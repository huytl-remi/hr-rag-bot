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
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bot import Bot

load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d'
)
logger = logging.getLogger(__name__)

# Load env variables with validation
required_vars = {
    "SLACK_BOT_TOKEN": os.getenv("SLACK_BOT_TOKEN"),
    "SLACK_SIGNING_SECRET": os.getenv("SLACK_SIGNING_SECRET"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "OPENAI_ASSISTANT_ID": os.getenv("OPENAI_ASSISTANT_ID")
}

if missing_vars := [var for var, value in required_vars.items() if not value]:
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
    try:
        logger.info("Starting up...")
        app.state.metrics = AppState()
        app.state.metrics.bot = Bot(
            api_key=str(required_vars["OPENAI_API_KEY"]),
            assistant_id=str(required_vars["OPENAI_ASSISTANT_ID"])
        )
        yield
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    finally:
        logger.info("Shutting down...")
        if hasattr(app.state.metrics, 'bot') and app.state.metrics.bot:
            await app.state.metrics.bot.cleanup()

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
slack_app = AsyncApp(
    token=required_vars["SLACK_BOT_TOKEN"],
    signing_secret=required_vars["SLACK_SIGNING_SECRET"]
)
handler = AsyncSlackRequestHandler(slack_app)

def get_confidence_emoji(confidence: str) -> str:
    return {
        "high": "🟢",
        "medium": "🟡",
        "low": "🔴"
    }.get(confidence.lower(), "❓")

def format_response_blocks(response_data: Dict[str, Any]) -> list:
    try:
        response = response_data.get("response", {})
        metadata = response_data.get("metadata", {})
        blocks = []

        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": (f"⏱️ Processed in {response.get('processing_time', 'N/A'):.2f}s | "
                        f"🌐 Lang: {metadata.get('language', 'unknown')}")
            }]
        })

        confidence = response.get("confidence", "low")
        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": f"{get_confidence_emoji(confidence)} Confidence: {confidence.upper()}"
            }]
        })

        answer = response.get("answer", "No answer provided")
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": answer
            }
        })

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
    logger.info(f"Received message event type: {event.get('type')}")

    try:
        channel_id: str = str(event.get("channel", ""))
        user_id: str = str(event.get("user", ""))

        if not channel_id or not user_id:
            logger.error("Missing channel_id or user_id")
            return

        if any([
            event.get("subtype") is not None,
            event.get("thread_ts") is not None,
            event.get("channel_type") != "im"
        ]):
            return

        app.state.metrics.request_count += 1

        if not app.state.metrics.bot:
            raise ValueError("Bot not initialized")

        # Get response from bot with all the new bells and whistles
        response = await app.state.metrics.bot.get_answer(
            query=event.get("text", "").strip(),
            user_id=user_id,
            say_func=say,
            channel_id=channel_id
        )

        blocks = format_response_blocks(response)

        # Send final response
        await say({
            "channel": channel_id,
            "text": response.get("response", {}).get("answer", "Processing your request..."),
            "blocks": blocks
        })

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        app.state.metrics.error_count += 1
        app.state.metrics.last_error = str(e)

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

    # Get metrics from bot if available
    bot_metrics = {}
    if app.state.metrics.bot and app.state.metrics.bot.metrics:
        bot_metrics = {
            "total_interactions": sum(len(interactions) for interactions in
                                   app.state.metrics.bot.metrics.interactions.values()),
            "unique_users": len(app.state.metrics.bot.metrics.interactions),
            "avg_response_time": (sum(app.state.metrics.bot.metrics.response_times) /
                                len(app.state.metrics.bot.metrics.response_times)
                                if app.state.metrics.bot.metrics.response_times else 0),
            "error_count": len(app.state.metrics.bot.metrics.errors)
        }

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "production"),
        "app_metrics": {
            "uptime_seconds": uptime,
            "request_count": app.state.metrics.request_count,
            "error_count": app.state.metrics.error_count,
            "error_rate": (app.state.metrics.error_count / app.state.metrics.request_count
                          if app.state.metrics.request_count > 0 else 0),
            "last_error": app.state.metrics.last_error
        },
        "bot_metrics": bot_metrics
    }

@app.get("/metrics/report")
async def get_metrics_report(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Generate a metrics report for HR."""
    try:
        if not app.state.metrics.bot:
            raise ValueError("Bot not initialized")

        start = (datetime.fromisoformat(start_date)
                if start_date else datetime.now() - timedelta(days=30))
        end = datetime.fromisoformat(end_date) if end_date else datetime.now()

        metrics = app.state.metrics.bot.metrics
        filtered_interactions = {
            user_id: [
                interaction for interaction in interactions
                if start <= interaction["timestamp"] <= end
            ]
            for user_id, interactions in metrics.interactions.items()
        }

        return {
            "period": {
                "start": start.isoformat(),
                "end": end.isoformat()
            },
            "usage_metrics": {
                "total_users": len(filtered_interactions),
                "total_queries": sum(len(interactions)
                                   for interactions in filtered_interactions.values()),
                "active_users": len([uid for uid, interactions in filtered_interactions.items()
                                   if interactions]),
                "avg_queries_per_user": (sum(len(interactions)
                                           for interactions in filtered_interactions.values()) /
                                       len(filtered_interactions) if filtered_interactions else 0)
            },
            "performance_metrics": {
                "avg_response_time": (sum(metrics.response_times) /
                                    len(metrics.response_times) if metrics.response_times else 0),
                "error_rate": (len(metrics.errors) /
                             len(metrics.questions) if metrics.questions else 0)
            },
            "questions": [
                {
                    "timestamp": q["timestamp"].isoformat(),
                    "user_id": q["user_id"],
                    "question": q["question"]
                }
                for q in metrics.questions
                if start <= q["timestamp"] <= end
            ]
        }
    except Exception as e:
        logger.error(f"Error generating metrics report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/slack/events")
async def endpoint(request: Request):
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
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
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
