# Lab 7 — Build a Client with an AI Coding Agent

[Sync your fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork#syncing-a-fork-branch-from-the-command-line) regularly — the lab gets updated.

## Product brief

> Build a Telegram bot that lets users interact with the LMS backend through chat. Users should be able to check system health, browse labs and scores, and ask questions in plain language. The bot should use an LLM to understand what the user wants and fetch the right data. Deploy it alongside the existing backend on the VM.

This is what a customer might tell you. Your job is to turn it into a working product using an AI coding agent (Qwen Code) as your development partner.

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  ┌──────────────┐     ┌──────────────────────────────────┐   │
│  │  Telegram    │────▶│  Your Bot                        │   │
│  │  User        │◀────│  (aiogram / python-telegram-bot) │   │
│  └──────────────┘     └──────┬───────────────────────────┘   │
│                              │                               │
│                              │ slash commands + plain text    │
│                              ├───────▶ /start, /help         │
│                              ├───────▶ /health, /labs        │
│                              ├───────▶ intent router ──▶ LLM │
│                              │                    │          │
│                              │                    ▼          │
│  ┌──────────────┐     ┌──────┴───────┐    tools/actions      │
│  │  Docker      │     │  LMS Backend │◀───── GET /items      │
│  │  Compose     │     │  (FastAPI)   │◀───── GET /analytics  │
│  │              │     │  + PostgreSQL│◀───── POST /sync      │
│  └──────────────┘     └──────────────┘                       │
└──────────────────────────────────────────────────────────────┘
```

## Requirements

### P0 — Must have

1. Testable handler architecture — handlers work without Telegram
2. CLI test mode: `cd bot && uv run bot.py --test "/command"` prints response to stdout
3. `/start` — welcome message
4. `/help` — lists all available commands
5. `/health` — calls backend, reports up/down status
6. `/labs` — lists available labs
7. `/scores <lab>` — per-task pass rates
8. Error handling — backend down produces a friendly message, not a crash

### P1 — Should have

1. Natural language intent routing — plain text interpreted by LLM
2. All 9 backend endpoints wrapped as LLM tools
3. Inline keyboard buttons for common actions
4. Multi-step reasoning (LLM chains multiple API calls)

### P2 — Nice to have

1. Rich formatting (tables, charts as images)
2. Response caching
3. Conversation context (multi-turn)

### P3 — Deployment

1. Bot containerized with Dockerfile
2. Added as service in `docker-compose.yml`
3. Deployed and running on VM
4. README documents deployment

## Learning advice

Notice the progression above: **product brief** (vague customer ask) → **prioritized requirements** (structured) → **task specifications** (precise deliverables + acceptance criteria). This is how engineering work flows.

You are not following step-by-step instructions — you are building a product with an AI coding agent. The learning comes from planning, building, testing, and debugging iteratively.

## Learning outcomes

By the end of this lab, you should be able to say:

1. I turned a vague product brief into a working Telegram bot.
2. I can ask it questions in plain language and it fetches the right data.
3. I used an AI coding agent to plan and build the whole thing.

## Tasks

### Prerequisites

1. Complete the [lab setup](./lab/setup/setup-simple.md#lab-setup)

> **Note**: First time in this course? Do the [full setup](./lab/setup/setup-full.md#lab-setup) instead.

### Required

1. [Plan and Scaffold](./lab/tasks/required/task-1.md) — P0: project structure + `--test` mode
2. [Backend Integration](./lab/tasks/required/task-2.md) — P0: slash commands + real data
3. [Intent-Based Natural Language Routing](./lab/tasks/required/task-3.md) — P1: LLM tool use
4. [Containerize and Document](./lab/tasks/required/task-4.md) — P3: containerize + deploy

## Deploy

This section documents how to deploy the bot on your VM using Docker.

### Prerequisites

- VM with Docker and docker-compose installed
- `.env.docker.secret` file with required environment variables
- Bot token from @BotFather
- LLM API key from Qwen Code

### Environment variables

Edit `.env.docker.secret` on your VM and set:

```bash
# Telegram Bot Token (from @BotFather)
BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyz

# LLM API credentials (from Qwen Code)
LLM_API_KEY=your-qwen-api-key
LLM_API_MODEL=qwen3-coder-flash
```

The bot also needs `LMS_API_KEY` which should already be set from the backend setup.

### Deploy commands

1. **Stop the background bot process** (if running):
   ```bash
   cd ~/se-toolkit-lab-7
   pkill -f "bot.py" 2>/dev/null || true
   ```

2. **Build and start all services**:
   ```bash
   docker compose --env-file .env.docker.secret up --build -d
   ```

3. **Verify all containers are running**:
   ```bash
   docker compose --env-file .env.docker.secret ps
   ```
   
   You should see:
   ```
   SERVICE    STATUS
   backend    Up
   bot        Up
   caddy      Up
   pgadmin    Up
   postgres   Up (healthy)
   ```

4. **Check bot logs**:
   ```bash
   docker compose --env-file .env.docker.secret logs bot --tail 20
   ```
   
   Look for:
   - "Starting bot... (polling)" — bot started successfully
   - "HTTP Request: POST .../getUpdates" — bot is polling Telegram
   - No Python tracebacks

### Verify in Telegram

Send these commands to your bot in Telegram:

1. `/start` — should return welcome message
2. `/health` — should show backend status
3. "what labs are available?" — should list labs (LLM-powered)
4. "which lab has the lowest pass rate?" — should compare and answer

### Troubleshooting

| Symptom | Solution |
|---------|----------|
| Bot container keeps restarting | Check logs: `docker compose logs bot`. Usually missing env var or import error. |
| `/health` fails | `LMS_API_URL` must be `http://backend:8000` (Docker service name, not localhost). |
| LLM queries fail | `LLM_API_BASE_URL` must use `http://host.docker.internal:42005/v1` — the Qwen proxy is on a different network. |
| "BOT_TOKEN is required" | Add `BOT_TOKEN` to `.env.docker.secret` and rebuild. |
| Build fails at `uv sync` | Make sure `bot/uv.lock` exists and is copied in Dockerfile. |

### Update deployment

After making code changes:

```bash
cd ~/se-toolkit-lab-7
git pull
docker compose --env-file .env.docker.secret up --build -d bot
```

This rebuilds only the bot container with the latest code.
