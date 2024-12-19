# HR Assistant Bot Deployment Guide

## Project Structure
```
hr-assistant-bot/
├── app.py             # Main FastAPI application
├── bot.py             # Bot implementation
├── requirements.txt   # Python dependencies
├── .env              # Environment variables (do not commit this)
└── DEPLOYMENT.md     # This guide
```

## Prerequisites
- Python 3.11 or higher
- Docker (if using containerized deployment)
- Network access to Slack and OpenAI APIs
- Proper environment variables configured

## Required Environment Variables
These must be set in the deployment environment:
```
# Slack Configuration
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret

# OpenAI Configuration
OPENAI_API_KEY=your-openai-key
OPENAI_ASSISTANT_ID=your-assistant-id

# App Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
PORT=8000
HOST=0.0.0.0
```

## Deployment Options

### Option 1: Direct Python Deployment
1. Install Python 3.11+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables
4. Run the application:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

### Option 2: Docker Deployment
1. Build the Docker image:
   ```bash
   docker build -t hr-assistant-bot .
   ```
2. Run the container:
   ```bash
   docker run -d \
     --name hr-assistant-bot \
     --env-file .env \
     -p 8000:8000 \
     hr-assistant-bot
   ```

## Dependencies
Make sure requirements.txt includes:
```
fastapi
uvicorn
slack-bolt
openai
python-dotenv
tenacity
```

## Post-Deployment Verification

1. Health Check
   - Access `https://your-domain/health`
   - Should return status "healthy"

2. Slack Configuration
   - After deployment, the URL needs to be added to Slack:
   - Add `https://your-domain/slack/events` to Event Subscriptions in Slack App settings

3. Testing
   - Send a DM to the bot in Slack
   - Monitor application logs for any errors
   - Check response time is acceptable
