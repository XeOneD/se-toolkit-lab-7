# Bot Development Plan

This document describes the development plan for the LMS Telegram bot across all four tasks.

## Overview

The bot integrates with the LMS backend to provide students with access to their lab progress, scores, and analytics via Telegram. The architecture follows the **separation of concerns** pattern: handlers are pure functions that don't depend on Telegram, making them testable via `--test` mode and reusable in unit tests.

## Task 1: Scaffold and Plan

**Goal:** Create the project structure and test mode.

- Create `bot/` directory with `bot.py` entry point
- Implement `--test` mode for offline testing
- Create handler layer (`handlers/`) with placeholder functions
- Set up configuration loading from `.env.bot.secret`
- Document the architecture in `PLAN.md`

**Key concept:** Testable handlers. The same handler function works from `--test` mode, unit tests, or Telegram.

## Task 2: Backend Integration

**Goal:** Connect handlers to the LMS backend API.

- Create `services/api_client.py` for HTTP requests
- Implement Bearer token authentication
- Update `/health` to check backend status
- Implement `/labs` to fetch available labs
- Implement `/scores` to fetch student scores
- Handle API errors gracefully (timeouts, 401, 500)

**Key concept:** API client pattern. Centralize HTTP logic in services, not handlers.

## Task 3: Intent Routing with LLM

**Goal:** Add natural language understanding using LLM tool calling.

- Create `services/llm_client.py` for LLM API calls
- Define tool descriptions for each handler
- Implement intent router that uses LLM to decide which tool to call
- Handle plain text queries like "what labs are available?"
- Improve tool descriptions for better LLM accuracy

**Key concept:** LLM tool use. The LLM reads tool descriptions to decide which function to call. Description quality matters more than prompt engineering.

## Task 4: Docker Deployment

**Goal:** Deploy the bot in Docker alongside the backend.

- Create `bot/Dockerfile` for the bot service
- Update `docker-compose.yml` to include the bot
- Configure Docker networking (containers use service names, not localhost)
- Set up environment variables for the bot container
- Test the deployed bot in Telegram

**Key concept:** Docker networking. Containers communicate via service names (`http://backend:8000`), not `localhost`.

## Testing Strategy

1. **Test mode:** `uv run bot.py --test "/command"` for quick verification
2. **Unit tests:** Test handlers in isolation (future)
3. **Manual testing:** Deploy to VM and test in Telegram

## File Structure

```
bot/
├── bot.py              # Entry point with --test mode
├── config.py           # Environment configuration
├── pyproject.toml      # Dependencies
├── PLAN.md             # This file
├── .env.bot.example    # Template for environment variables
├── handlers/
│   └── __init__.py     # Command handlers (no Telegram dependency)
└── services/
    ├── __init__.py     # API and LLM clients
```
