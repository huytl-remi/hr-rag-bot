from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from contextlib import asynccontextmanager
import os
import logging
import time
from datetime import datetime
from dotenv import load_dotenv
from bot import Bot

load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d'
)
logger = logging.getLogger(__name__)

# Load env variables
bot_token = os.getenv("SLACK_BOT_TOKEN")
signing_secret = os.getenv("SLACK_SIGNING_SECRET")
openai_key = os.getenv("OPENAI_API_KEY")
assistant_id = os.getenv("OPENAI_ASSISTANT_ID")

# Validate required env vars
required_vars = {
    "SLACK_BOT_TOKEN": bot_token,
    "SLACK_SIGNING_SECRET": signing_secret,
    "OPENAI_API_KEY": openai_key,
    "OPENAI_ASSISTANT_ID": assistant_id
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

class AppState:
    def __init__(self):
        self.start_time: float = time.time()
        self.request_count: int = 0
        self.error_count: int = 0
        self.last_error: Optional[str] = None
        self.bot: Optional[Bot] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Proper async initialization of bot with type checking"""
    try:
        # Startup
        logger.info("Starting up...")
        app.state.metrics = AppState()

        # Add type checking for env vars
        if not isinstance(openai_key, str) or not isinstance(assistant_id, str):
            raise ValueError("API key and assistant ID must be strings")

        app.state.metrics.bot = Bot(
            api_key=str(openai_key),
            assistant_id=str(assistant_id)
        )
        yield
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down...")
        if hasattr(app.state.metrics, 'bot') and app.state.metrics.bot:
            await app.state.metrics.bot.cleanup()

# Initialize FastAPI with lifecycle management
app = FastAPI(
    title="HR Assistant Bot",
    description="Slack bot for HR assistance using OpenAI",
    version="1.0.0",
    lifespan=lifespan
)

# Add middlewares
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("ALLOWED_ORIGINS", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Slack app
slack_app = AsyncApp(token=bot_token, signing_secret=signing_secret)
handler = AsyncSlackRequestHandler(slack_app)

def get_confidence_emoji(confidence: str) -> str:
    """Get emoji indicator for confidence level."""
    return {
        "high": "🟢",
        "medium": "🟡",
        "low": "🔴"
    }.get(confidence.lower(), "❓")

def format_response_blocks(response_data: Dict[str, Any]) -> list:
    """Format JSON response into Slack blocks with error handling."""
    try:
        response = response_data.get("response", {})
        metadata = response_data.get("metadata", {})
        blocks = []

        # Processing time and metadata
        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": (f"⏱️ Processed in {response.get('processing_time', 'N/A')}s | "
                        f"🌐 Lang: {metadata.get('language', 'unknown')}")
            }]
        })

        # Confidence indicator
        confidence = response.get("confidence", "low")
        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": f"{get_confidence_emoji(confidence)} Confidence: {confidence.upper()}"
            }]
        })

        # Main answer
        answer = response.get("answer", "No answer provided")
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": answer
            }
        })

        # Sources
        sources = response.get("sources", [])
        if sources:
            blocks.append({"type": "divider"})
            source_text = "*Sources:*\n" + "\n".join([f"• {source}" for source in sources])
            blocks.append({
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": source_text
                }]
            })

        return blocks
    except Exception as e:
        logger.error(f"Error formatting response: {str(e)}", exc_info=True)
        return [{
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Error formatting response. Please try again."
            }
        }]

@slack_app.event("message")
async def handle_message(event, say):
    """Handle incoming DM messages with proper type checking."""
    logger.info(f"Received message event type: {event.get('type')}")

    try:
        # Message validation with type checking
        channel_id: str = str(event.get("channel", ""))
        user_id: str = str(event.get("user", ""))

        if not channel_id or not user_id:
            logger.error("Missing channel_id or user_id")
            return

        # Various checks
        if any([
            event.get("subtype") is not None,
            event.get("thread_ts") is not None,
            event.get("channel_type") != "im"
        ]):
            return

        # Send typing indicator
        try:
            await say({
                "channel": channel_id,
                "text": "Processing your request...",
                "blocks": [{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "🤔 Processing your request..."
                    }
                }]
            })
        except Exception as e:
            logger.error(f"Failed to send typing indicator: {str(e)}")
            return

        # Get response from bot
        if not app.state.metrics.bot:
            raise ValueError("Bot not initialized")

        start_time = time.time()
        response = await app.state.metrics.bot.get_answer(event.get("text", "").strip())
        processing_time = time.time() - start_time

        # Add processing time if not present
        if "response" in response:
            response["response"]["processing_time"] = processing_time

        blocks = format_response_blocks(response)

        # Send response
        await say({
            "channel": channel_id,
            "text": response.get("response", {}).get("answer", "Processing your request..."),
            "blocks": blocks
        })

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        error_block = [{
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Sorry, I encountered an error. Please try again later."
            }
        }]

        if channel_id:
            try:
                await say({
                    "channel": channel_id,
                    "text": "Error processing request",
                    "blocks": error_block
                })
            except Exception as e2:
                logger.error(f"Failed to send error message: {str(e2)}")

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with metrics."""
    uptime = time.time() - app.state.metrics.start_time
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "production"),
        "metrics": {
            "uptime_seconds": uptime,
            "request_count": app.state.metrics.request_count,
            "error_count": app.state.metrics.error_count,
            "error_rate": (app.state.metrics.error_count / app.state.metrics.request_count
                          if app.state.metrics.request_count > 0 else 0),
            "last_error": app.state.metrics.last_error
        }
    }

@app.post("/slack/events")
async def endpoint(request: Request):
    """Slack events endpoint with enhanced error handling."""
    try:
        body = await request.json()
        if body.get("type") == "url_verification":
            logger.info("Handling Slack URL verification challenge")
            return {"challenge": body.get("challenge")}

        logger.info(f"Processing slack event type: {body.get('type')}")
        return await handler.handle(request)
    except Exception as e:
        logger.error(f"Error processing slack event: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        reload=os.getenv("ENVIRONMENT") == "development"
    )
