# Bot Dockerfile - builds the Telegram bot using uv
# Adapted from backend/Dockerfile

ARG REGISTRY_PREFIX=harbor.pg.innopolis.university/docker-hub-cache/
FROM ${REGISTRY_PREFIX}astral/uv:python3.14-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_NO_DEV=1
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Copy bot-specific files
COPY bot/pyproject.toml ./
COPY bot/uv.lock ./

# Sync dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# Copy bot source code
COPY bot/ ./bot/

# ---
# Stage 2: Final image
# ---

ARG REGISTRY_PREFIX=harbor.pg.innopolis.university/docker-hub-cache/
FROM ${REGISTRY_PREFIX}python:3.14.2-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

# Setup non-root user
RUN groupadd --system --gid 999 nonroot \
    && useradd --system --gid 999 --uid 999 --create-home nonroot

# Copy application from builder
COPY --from=builder --chown=nonroot:nonroot /app /app

# Add uv environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Use non-root user
USER nonroot

WORKDIR /app

# Run the bot
CMD ["python", "bot/bot.py"]
