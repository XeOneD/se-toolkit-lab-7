"""Command handlers for the Telegram bot.

Handlers are pure functions that take input and return text.
They don't depend on Telegram - same logic works from --test mode,
unit tests, or Telegram handler.
"""

from typing import Any


async def handle_start(args: list[str]) -> str:
    """Handle /start command.
    
    Args:
        args: Command arguments (not used for /start)
    
    Returns:
        Welcome message
    """
    return "Welcome to LMS Bot! Use /help to see available commands."


async def handle_help(args: list[str]) -> str:
    """Handle /help command.
    
    Args:
        args: Command arguments (not used for /help)
    
    Returns:
        List of available commands
    """
    return """Available commands:
/start - Welcome message
/help - Show this help message
/health - Check backend status
/labs - List available labs
/scores - View your scores"""


async def handle_health(args: list[str]) -> str:
    """Handle /health command.
    
    Args:
        args: Command arguments (not used for /health)
    
    Returns:
        Backend health status (placeholder)
    """
    return "Backend status: OK (placeholder)"


async def handle_labs(args: list[str]) -> str:
    """Handle /labs command.
    
    Args:
        args: Command arguments (not used for /labs)
    
    Returns:
        List of available labs (placeholder)
    """
    return "Available labs will be shown here (placeholder)"


async def handle_scores(args: list[str]) -> str:
    """Handle /scores command.
    
    Args:
        args: Command arguments (e.g., lab name)
    
    Returns:
        Score information (placeholder)
    """
    return f"Scores for {' '.join(args) if args else 'all labs'} (placeholder)"


async def handle_unknown(text: str) -> str:
    """Handle unknown commands or plain text.
    
    Args:
        text: The input text
    
    Returns:
        Response for unknown input
    """
    return f"I don't understand: {text}. Use /help for available commands."
