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
1. Test mode: uv run bot.py --test "query" - prints response to stdout
2. Telegram mode: uv run bot.py - connects to Telegram and handles messages
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
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


async def process_command(
    text: str,
    api_client=None,
    intent_router=None,
) -> str:
    """Process a command and return response.

    Args:
        text: Input text (command with optional arguments)
        api_client: LMS API client
        intent_router: Intent router for natural language queries

    Returns:
        Response text
    """
    command, args = parse_command(text)

    # Handle slash commands
    if command == "/start":
        return "Welcome to LMS Bot! Use /help to see available commands."
    elif command == "/help":
        return """Available commands:
/start - Welcome message
/help - Show this help message
/health - Check backend status
/labs - List available labs
/scores <lab> - View scores for a specific lab (e.g., /scores lab-04)

You can also ask questions in natural language:
- "What labs are available?"
- "Show me scores for lab 4"
- "Which lab has the lowest pass rate?"
- "Who are the top 5 students?"
"""
    elif command == "/health":
        if api_client:
            try:
                result = await api_client.health_check()
                return f"Backend is healthy. {result['items_count']} items available."
            except Exception as e:
                return f"Backend error: {type(e).__name__}. {str(e)[:100]}"
        return "Backend status: OK (API client not initialized)"
    elif command == "/labs":
        if api_client:
            try:
                items = await api_client.get_items()
                labs = [item for item in items if item.get("type") == "lab"]
                if not labs:
                    return "No labs available."
                result = ["Available labs:"]
                for lab in labs:
                    result.append(f"- {lab.get('title', 'Unknown')}")
                return "\n".join(result)
            except Exception as e:
                return f"Backend error: {type(e).__name__}. {str(e)[:100]}"
        return "Labs data not available (API client not initialized)"
    elif command == "/scores":
        if api_client:
            if not args:
                return "Please specify a lab. Usage: /scores <lab> (e.g., /scores lab-04)"
            try:
                data = await api_client.get_analytics_pass_rates(args[0])
                if isinstance(data, dict):
                    pass_rates = data.get("pass_rates", [])
                else:
                    pass_rates = data
                if not pass_rates:
                    return f"No data available for {args[0]}."
                result = [f"Pass rates for {args[0]}:"]
                for rate in pass_rates:
                    task = rate.get("task", "Unknown")
                    score = rate.get("avg_score") or rate.get("pass_rate", 0)
                    attempts = rate.get("attempts", 0)
                    if rate.get("pass_rate") is not None:
                        percentage = score * 100
                    else:
                        percentage = score
                    result.append(f"- {task}: {percentage:.1f}% ({attempts} attempts)")
                return "\n".join(result)
            except Exception as e:
                return f"Backend error: {type(e).__name__}. {str(e)[:100]}"
        return "Scores data not available (API client not initialized)"
    
    # Natural language query - use intent router
    if intent_router:
        return await intent_router.route(text, debug=True)
    
    return f"I don't understand: {text}. Use /help for available commands."


def get_start_keyboard() -> InlineKeyboardMarkup:
    """Get inline keyboard for /start command."""
    keyboard = [
        [
            InlineKeyboardButton("📚 Available Labs", callback_data="labs"),
            InlineKeyboardButton("🏥 Health Check", callback_data="health"),
        ],
        [
            InlineKeyboardButton("📊 Lab Scores", callback_data="scores_help"),
            InlineKeyboardButton("❓ Help", callback_data="help"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


async def button_callback_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    api_client,
    intent_router,
) -> None:
    """Handle inline keyboard button callbacks."""
    query = update.callback_query
    if query is None:
        return
    
    await query.answer()
    
    data = query.data
    if data == "labs":
        response = await process_command("/labs", api_client, intent_router)
    elif data == "health":
        response = await process_command("/health", api_client, intent_router)
    elif data == "scores_help":
        response = "To view scores, use /scores <lab> (e.g., /scores lab-04) or ask me 'show me scores for lab 4'"
    elif data == "help":
        response = await process_command("/help", api_client, intent_router)
    else:
        response = "Unknown action"
    
    await query.edit_message_text(response)


def main() -> None:
    """Main entry point."""
    import asyncio

    # Check for --test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        if len(sys.argv) < 3:
            print("Usage: uv run bot.py --test <query>", file=sys.stderr)
            print("Example: uv run bot.py --test 'what labs are available'", file=sys.stderr)
            sys.exit(1)

        # Test mode: run command and print response
        query = sys.argv[2]

        # Initialize API client and intent router
        from config import settings
        from services.api_client import LMSAPIClient
        from services.intent_router import IntentRouter

        api_client = LMSAPIClient(settings.lms_api_url, settings.lms_api_key)
        intent_router = IntentRouter(
            api_client,
            settings.llm_api_key,
            settings.llm_api_base_url,
            settings.llm_api_model,
        )

        response = asyncio.run(process_command(query, api_client, intent_router))
        print(response)
        sys.exit(0)

    # Telegram mode: start the bot
    from config import settings
    from services.api_client import LMSAPIClient
    from services.intent_router import IntentRouter

    if not settings.bot_token:
        print("Error: BOT_TOKEN not set in .env.bot.secret", file=sys.stderr)
        sys.exit(1)

    # Initialize API client and intent router
    api_client = LMSAPIClient(settings.lms_api_url, settings.lms_api_key)
    intent_router = IntentRouter(
        api_client,
        settings.llm_api_key,
        settings.llm_api_base_url,
        settings.llm_api_model,
    )

    async def telegram_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle Telegram messages."""
        if update.message is None:
            return

        user_input = update.message.text
        if not user_input:
            return

        response = await process_command(user_input, api_client, intent_router)
        await update.message.reply_text(response)

    async def start_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command with inline keyboard."""
        if update.message is None:
            return
        
        response = await process_command("/start", api_client, intent_router)
        await update.message.reply_text(
            response,
            reply_markup=get_start_keyboard(),
        )

    async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle inline keyboard button clicks."""
        await button_callback_handler(update, context, api_client, intent_router)

    # Create application
    application = Application.builder().token(settings.bot_token).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start_command_handler))
    application.add_handler(CommandHandler("help", telegram_handler))
    application.add_handler(CommandHandler("health", telegram_handler))
    application.add_handler(CommandHandler("labs", telegram_handler))
    application.add_handler(CommandHandler("scores", telegram_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, telegram_handler))
    application.add_handler(CallbackQueryHandler(callback_query_handler))

    # Start the bot
    print(f"Starting bot... (polling)")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
