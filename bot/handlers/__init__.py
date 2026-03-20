"""Command handlers for the Telegram bot.

Handlers are pure functions that take input and return text.
They don't depend on Telegram - same logic works from --test mode,
unit tests, or Telegram handler.
"""

from typing import Any

import httpx

from services.api_client import LMSAPIClient


# Global API client instance (initialized from config)
_api_client: LMSAPIClient | None = None


def init_api_client(base_url: str, api_key: str) -> None:
    """Initialize the LMS API client.
    
    Args:
        base_url: Base URL of the LMS backend
        api_key: API key for authentication
    """
    global _api_client
    _api_client = LMSAPIClient(base_url, api_key)


def _get_api_client() -> LMSAPIClient:
    """Get the API client, raising an error if not initialized."""
    if _api_client is None:
        raise RuntimeError(
            "API client not initialized. Call init_api_client() first."
        )
    return _api_client


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
/scores <lab> - View scores for a specific lab (e.g., /scores lab-04)"""


async def handle_health(args: list[str]) -> str:
    """Handle /health command.
    
    Args:
        args: Command arguments (not used for /health)
    
    Returns:
        Backend health status
    """
    try:
        client = _get_api_client()
        result = await client.health_check()
        return f"Backend is healthy. {result['items_count']} items available."
    except httpx.ConnectError as e:
        return f"Backend error: connection refused. Check that the services are running."
    except httpx.HTTPStatusError as e:
        return f"Backend error: HTTP {e.response.status_code} {e.response.reason_phrase}. The backend service may be down."
    except httpx.HTTPError as e:
        return f"Backend error: {type(e).__name__}. {str(e)[:100]}"
    except Exception as e:
        return f"Backend error: {type(e).__name__}. {str(e)[:100]}"


async def handle_labs(args: list[str]) -> str:
    """Handle /labs command.
    
    Args:
        args: Command arguments (not used for /labs)
    
    Returns:
        List of available labs
    """
    try:
        client = _get_api_client()
        items = await client.get_items()
        
        # Filter only labs (type="lab")
        labs = [item for item in items if item.get("type") == "lab"]
        
        if not labs:
            return "No labs available."
        
        result = ["Available labs:"]
        for lab in labs:
            title = lab.get("title", "Unknown")
            result.append(f"- {title}")
        
        return "\n".join(result)
    except httpx.ConnectError as e:
        return f"Backend error: connection refused. Check that the services are running."
    except httpx.HTTPStatusError as e:
        return f"Backend error: HTTP {e.response.status_code} {e.response.reason_phrase}. The backend service may be down."
    except httpx.HTTPError as e:
        return f"Backend error: {type(e).__name__}. {str(e)[:100]}"
    except Exception as e:
        return f"Backend error: {type(e).__name__}. {str(e)[:100]}"


async def handle_scores(args: list[str]) -> str:
    """Handle /scores command.
    
    Args:
        args: Command arguments (lab name, e.g., ["lab-04"])
    
    Returns:
        Score information for the specified lab
    """
    if not args:
        return "Please specify a lab. Usage: /scores <lab> (e.g., /scores lab-04)"
    
    lab_name = args[0].lower()
    
    try:
        client = _get_api_client()
        data = await client.get_analytics_pass_rates(lab_name)
        
        # The response format is a list: [{"task": "...", "avg_score": 60.9, "attempts": 686}, ...]
        # Handle both list and dict formats
        if isinstance(data, dict):
            pass_rates = data.get("pass_rates", [])
        else:
            pass_rates = data  # It's already a list
        
        if not pass_rates:
            return f"No data available for {lab_name}."
        
        result = [f"Pass rates for {lab_name}:"]
        for rate in pass_rates:
            task = rate.get("task", "Unknown task")
            # Handle both avg_score and pass_rate formats
            score = rate.get("avg_score") or rate.get("pass_rate", 0)
            attempts = rate.get("attempts", 0)
            # avg_score is already a percentage, pass_rate is 0-1
            if rate.get("pass_rate") is not None:
                percentage = score * 100
            else:
                percentage = score
            result.append(f"- {task}: {percentage:.1f}% ({attempts} attempts)")
        
        return "\n".join(result)
    except httpx.ConnectError as e:
        return f"Backend error: connection refused. Check that the services are running."
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"Lab '{lab_name}' not found. Use /labs to see available labs."
        return f"Backend error: HTTP {e.response.status_code} {e.response.reason_phrase}. The backend service may be down."
    except httpx.HTTPError as e:
        return f"Backend error: {type(e).__name__}. {str(e)[:100]}"
    except Exception as e:
        return f"Backend error: {type(e).__name__}. {str(e)[:100]}"


async def handle_unknown(text: str) -> str:
    """Handle unknown commands or plain text.
    
    Args:
        text: The input text
    
    Returns:
        Response for unknown input
    """
    return f"I don't understand: {text}. Use /help for available commands."
