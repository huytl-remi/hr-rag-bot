from typing import Dict, Any, Optional, Set
from openai import OpenAI, AsyncOpenAI
import logging
import json
import re
import asyncio
from datetime import datetime, timedelta
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, calls: int, period: int):
        self.calls = calls  # number of calls allowed
        self.period = period  # in seconds
        self.timestamps = []

    async def acquire(self):
        now = datetime.now()
        # Remove timestamps older than our period
        self.timestamps = [ts for ts in self.timestamps
                         if now - ts < timedelta(seconds=self.period)]

        if len(self.timestamps) >= self.calls:
            sleep_time = (self.timestamps[0] +
                         timedelta(seconds=self.period) - now).total_seconds()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.timestamps.append(now)

class Bot:
    def __init__(self, api_key: str, assistant_id: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.assistant_id = assistant_id
        self.active_threads: Set[str] = set()
        # Rate limit to 50 requests per minute
        self.rate_limiter = RateLimiter(calls=50, period=60)

    def _extract_json_from_markdown(self, text: str) -> Dict:
        """Extract JSON from text, handling both raw JSON and markdown code blocks."""
        try:
            # First try direct JSON parsing
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract from markdown code block
            patterns = [
                r'```(?:json\s*)?\n?(.*?)```',  # code blocks
                r'{.*}',                         # raw json
            ]

            for pattern in patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                for match in matches:
                    try:
                        cleaned = match.strip()
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        continue

            logger.error(f"Failed to parse JSON from text: {text[:200]}...")
            raise ValueError("No valid JSON found in response")

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_answer(self, query: str) -> Dict[str, Any]:
        """Get answer using OpenAI assistant with retry logic."""
        try:
            # Acquire rate limiting token
            await self.rate_limiter.acquire()

            # Create thread
            thread = await self.client.beta.threads.create()
            self.active_threads.add(thread.id)

            try:
                # Add message to thread
                await self.client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=f"""Query: {query}

                    Detect the language and respond EXACTLY in this JSON format:
                    ```json
                    {{
                        "metadata": {{
                            "language": "en"  // ISO 639-1 code of query language
                        }},
                        "response": {{
                            "answer": "your answer in the SAME language as the query",
                            "sources": ["source1", "source2"],
                            "confidence": "high|medium|low"
                        }}
                    }}
                    ```"""
                )

                # Run assistant
                start_time = datetime.now()
                run = await self.client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=self.assistant_id
                )

                # Poll for completion
                timeout = 30  # seconds
                poll_interval = 1  # second
                elapsed = 0

                while elapsed < timeout:
                    run_status = await self.client.beta.threads.runs.retrieve(
                        thread_id=thread.id,
                        run_id=run.id
                    )

                    if run_status.status == "completed":
                        break
                    elif run_status.status == "failed":
                        raise Exception(f"Assistant run failed: {run_status.last_error}")

                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval

                if elapsed >= timeout:
                    raise TimeoutError("Assistant response timed out")

                messages = await self.client.beta.threads.messages.list(
                    thread_id=thread.id
                )

                # Process response
                for msg in messages.data:
                    if msg.role == "assistant":
                        for content in msg.content:
                            if content.type == 'text':
                                response_data = self._extract_json_from_markdown(content.text.value)
                                # Add processing time
                                response_data["response"]["processing_time"] = \
                                    (datetime.now() - start_time).total_seconds()
                                return response_data

                raise ValueError("No valid response content found")

            finally:
                # Cleanup thread
                try:
                    await self.client.beta.threads.delete(thread.id)
                    self.active_threads.remove(thread.id)
                except Exception as e:
                    logger.warning(f"Failed to delete thread {thread.id}: {str(e)}")

        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}", exc_info=True)
            return {
                "metadata": {
                    "language": "en",
                    "timestamp": datetime.now().isoformat(),
                    "query_length": len(query)
                },
                "response": {
                    "answer": f"Error: {str(e)}",
                    "sources": [],
                    "confidence": "low",
                    "processing_time": 0
                }
            }

    async def cleanup(self):
        """Cleanup any remaining threads."""
        for thread_id in list(self.active_threads):
            try:
                await self.client.beta.threads.delete(thread_id)
                self.active_threads.remove(thread_id)
            except Exception as e:
                logger.error(f"Failed to cleanup thread {thread_id}: {str(e)}")
