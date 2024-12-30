from typing import Dict, Any, Optional, Set, List, Tuple
from openai import AsyncOpenAI
import logging
import json
import re
import asyncio
import csv
import os
from datetime import datetime, timedelta
from collections import defaultdict
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self, report_channel: str = "phuongntn"):
        self.interactions = defaultdict(list)
        self.questions = []
        self.response_times = []
        self.errors = []
        self.report_channel = report_channel
        self.last_report_time = datetime.now()

    async def log_interaction(self, user_id: str, question: str,
                            response_time: float, confidence: str,
                            error: Optional[str] = None) -> Tuple[Optional[str], Optional[List[List[str]]], bool]:
        timestamp = datetime.now()

        # Log the interaction
        interaction = {
            "timestamp": timestamp,
            "question": question,
            "response_time": response_time,
            "confidence": confidence
        }

        if error:
            interaction["error"] = error
            self.errors.append({
                "timestamp": timestamp,
                "user_id": user_id,
                "error": error,
                "question": question
            })

        self.interactions[user_id].append(interaction)
        self.questions.append({
            "timestamp": timestamp,
            "user_id": user_id,
            "question": question
        })
        self.response_times.append(response_time)

        # Check if it's time for weekly report
        if (timestamp - self.last_report_time) >= timedelta(days=7):
            summary, csv_data = await self.generate_weekly_report()
            return summary, csv_data, True

        return None, None, False

    async def generate_weekly_report(self) -> Tuple[str, List[List[str]]]:
        """Generate weekly report data."""
        start_date = self.last_report_time
        end_date = datetime.now()

        csv_rows = [["Timestamp", "User", "Question", "Response Time", "Confidence", "Error"]]

        for user_id, interactions in self.interactions.items():
            for interaction in interactions:
                if start_date <= interaction["timestamp"] <= end_date:
                    csv_rows.append([
                        interaction["timestamp"].isoformat(),
                        user_id,
                        interaction["question"],
                        f"{interaction['response_time']:.2f}",
                        interaction["confidence"],
                        interaction.get("error", "")
                    ])

        total_queries = len([i for i in self.questions
                           if start_date <= i["timestamp"] <= end_date])
        unique_users = len(set(i["user_id"] for i in self.questions
                             if start_date <= i["timestamp"] <= end_date))
        avg_response = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        error_rate = len(self.errors) / total_queries if total_queries else 0

        summary_text = f"""Weekly Bot Report ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})

📊 Summary:
- Total Queries: {total_queries}
- Unique Users: {unique_users}
- Avg Response Time: {avg_response:.2f}s
- Error Rate: {error_rate:.2%}

Top 5 Most Asked Topics:
{self._get_top_topics()}

See attached CSV for detailed data."""

        self.last_report_time = end_date
        return summary_text, csv_rows

    def _get_top_topics(self, n: int = 5) -> str:
        """Get most common questions (basic implementation)."""
        recent_questions = [q["question"] for q in self.questions[-n:]]
        return "\n".join(f"• {q}" for q in recent_questions)

class RequestQueue:
    def __init__(self, max_retries: int = 3):
        self.queue = asyncio.Queue()
        self.max_retries = max_retries
        self.processing = False

    async def add_request(self, request_func, *args):
        await self.queue.put((request_func, args, 0))

        if not self.processing:
            self.processing = True
            asyncio.create_task(self.process_queue())

    async def process_queue(self):
        while not self.queue.empty():
            func, args, retries = await self.queue.get()
            try:
                await func(*args)
            except Exception as e:
                if retries < self.max_retries - 1:
                    await asyncio.sleep(2 ** retries)
                    await self.queue.put((func, args, retries + 1))
                else:
                    logger.error(f"Failed after {self.max_retries} retries: {e}")
        self.processing = False

class RateLimiter:
    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.timestamps = []

    async def acquire(self):
        now = datetime.now()
        self.timestamps = [ts for ts in self.timestamps
                         if now - ts < timedelta(seconds=self.period)]

        if len(self.timestamps) >= self.calls:
            sleep_time = (self.timestamps[0] +
                         timedelta(seconds=self.period) - now).total_seconds()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.timestamps.append(now)

class Bot:
    def __init__(self, api_key: str, assistant_id: str, slack_client: Any = None):
        self.client = AsyncOpenAI(api_key=api_key)
        self.assistant_id = assistant_id
        self.active_threads: Set[str] = set()
        self.rate_limiter = RateLimiter(calls=50, period=60)
        self.request_queue = RequestQueue()
        self.metrics = MetricsCollector()
        self.slack_client = slack_client

    async def update_status(self, say_func, channel_id: str, message_ts: Optional[str],
                          stage: str) -> Optional[str]:
        """Update or send new status message."""
        try:
            if message_ts:
                await say_func.client.chat_update(
                    channel=channel_id,
                    ts=message_ts,
                    text=stage,
                    blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": stage}}]
                )
                return message_ts
            else:
                response = await say_func({
                    "channel": channel_id,
                    "text": stage,
                    "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": stage}}]
                })
                return response['ts']
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
            return message_ts

    def _extract_json_from_markdown(self, text: str) -> Dict:
        """Extract JSON from text, handling both raw JSON and markdown code blocks."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            patterns = [
                r'```(?:json\s*)?\n?(.*?)```',
                r'{.*}',
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

    async def send_weekly_report(self, summary: str, csv_data: List[List[str]]):
        """Send weekly report to HR."""
        try:
            if not self.slack_client:
                logger.warning("Slack client not initialized, skipping report")
                return

            filename = f"bot_metrics_{datetime.now().strftime('%Y%m%d')}.csv"
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)

            await self.slack_client.files_upload(
                channels=self.metrics.report_channel,
                initial_comment=summary,
                file=filename,
                filename=filename,
                title="Weekly Bot Metrics Report"
            )

            os.remove(filename)

        except Exception as e:
            logger.error(f"Failed to send weekly report: {e}")

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_answer(self, query: str, user_id: str,
                        say_func: Any, channel_id: str) -> Dict[str, Any]:
        """Get answer using OpenAI assistant with status updates."""
        start_time = datetime.now()
        message_ts = None
        stages = [
            "🤔 Processing your request...",
            "📚 Reading through documents...",
            "💭 Analyzing the information..."
        ]

        try:
            await self.rate_limiter.acquire()

            thread = await self.client.beta.threads.create()
            self.active_threads.add(thread.id)

            try:
                for stage in stages:
                    message_ts = await self.update_status(say_func, channel_id,
                                                        message_ts, stage)
                    await asyncio.sleep(2.5)

                await self.client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=f"""Query: {query}

                    Be truthful. When you don't know, say so. When sources don't exist, acknowledge it in your response.

                    Detect the query's language and answer in that language.

                    Respond in this JSON format:
                    ```json
                    {{
                        "metadata": {{
                            "language": "en"  // detected query language
                        }},
                        "response": {{
                            "answer": "your response in query's language",
                            "sources": ["only include REAL sources you're certain about"],
                            "confidence": "high|medium|low"  // your confidence in the completeness and accuracy
                        }}
                    }}
                    ```"""
                )

                run = await self.client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=self.assistant_id
                )

                timeout = 30
                poll_interval = 1
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

                for msg in messages.data:
                    if msg.role == "assistant":
                        for content in msg.content:
                            if content.type == 'text':
                                # Clean up status message
                                if message_ts:
                                    try:
                                        await say_func.client.chat_delete(
                                            channel=channel_id,
                                            ts=message_ts
                                        )
                                    except Exception as e:
                                        logger.warning(f"Failed to delete status message: {e}")

                                response_data = self._extract_json_from_markdown(content.text.value)
                                processing_time = (datetime.now() - start_time).total_seconds()
                                response_data["response"]["processing_time"] = processing_time

                                # Log metrics and check if we need to send a report
                                summary, csv_data, should_send_report = await self.metrics.log_interaction(
                                    user_id=user_id,
                                    question=query,
                                    response_time=processing_time,
                                    confidence=response_data["response"]["confidence"]
                                )

                                if should_send_report and summary and csv_data:
                                    await self.send_weekly_report(summary, csv_data)

                                return response_data

                raise ValueError("No valid response content found")

            finally:
                try:
                    await self.client.beta.threads.delete(thread.id)
                    self.active_threads.remove(thread.id)  # was thread_id
                except Exception as e:
                    logger.warning(f"Failed to delete thread {thread.id}: {str(e)}")

        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}", exc_info=True)
            processing_time = (datetime.now() - start_time).total_seconds()

            await self.metrics.log_interaction(
                user_id=user_id,
                question=query,
                response_time=processing_time,
                confidence="low",
                error=str(e)
            )

            await self.request_queue.add_request(self.get_answer, query, user_id,
                                              say_func, channel_id)

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
                    "processing_time": processing_time
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
