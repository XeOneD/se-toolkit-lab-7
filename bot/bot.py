#!/usr/bin/env uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "httpx>=0.28.1",
#     "pydantic-settings>=2.12.0",
#     "python-telegram-bot>=21.0",
# ]
# ///

"""Telegram bot entry point.

Supports two modes:
1. Test mode: uv run bot.py --test "/command" - prints response to stdout
2. Telegram mode: uv run bot.py - connects to Telegram and handles messages
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from handlers import (
    handle_help,
    handle_health,
    handle_labs,
    handle_scores,
    handle_start,
    handle_unknown,
)


def parse_command(text: str) -> tuple[str, list[str]]:
    """Parse command text into command and arguments.
    
    Args:
        text: Input text (e.g., "/scores lab-04")
    
    Returns:
        Tuple of (command, args)
    """
    parts = text.strip().split()
    if not parts:
        return "", []
    
    command = parts[0].lower()
    args = parts[1:] if len(parts) > 1 else []
    return command, args


async def process_command(text: str) -> str:
    """Process a command and return response.
    
    Args:
        text: Input text (command with optional arguments)
    
    Returns:
        Response text
    """
    command, args = parse_command(text)
    
    if command == "/start":
        return await handle_start(args)
    elif command == "/help":
        return await handle_help(args)
    elif command == "/health":
        return await handle_health(args)
    elif command == "/labs":
        return await handle_labs(args)
    elif command == "/scores":
        return await handle_scores(args)
    else:
        return await handle_unknown(text)


def main() -> None:
    """Main entry point."""
    import asyncio

    # Check for --test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        if len(sys.argv) < 3:
            print("Usage: uv run bot.py --test <command>", file=sys.stderr)
            print("Example: uv run bot.py --test '/start'", file=sys.stderr)
            sys.exit(1)

        # Test mode: run command and print response
        command = sys.argv[2]
        response = asyncio.run(process_command(command))
        print(response)
        sys.exit(0)

    # Telegram mode: start the bot
    from telegram import Update
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )

    # Load configuration
    from config import settings

    if not settings.bot_token:
        print("Error: BOT_TOKEN not set in .env.bot.secret", file=sys.stderr)
        sys.exit(1)

    async def telegram_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle Telegram messages."""
        if update.message is None:
            return

        user_input = update.message.text
        if not user_input:
            return

        response = await process_command(user_input)
        await update.message.reply_text(response)

    # Create application
    application = Application.builder().token(settings.bot_token).build()

    # Add handlers
    application.add_handler(CommandHandler("start", telegram_handler))
    application.add_handler(CommandHandler("help", telegram_handler))
    application.add_handler(CommandHandler("health", telegram_handler))
    application.add_handler(CommandHandler("labs", telegram_handler))
    application.add_handler(CommandHandler("scores", telegram_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, telegram_handler))

    # Start the bot
    print(f"Starting bot... (polling)")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
