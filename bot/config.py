"""Configuration loader for the Telegram bot.

Reads environment variables from .env.bot.secret file.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    """Bot configuration."""

    bot_token: str = Field(default="", alias="BOT_TOKEN")
    lms_api_url: str = Field(default="http://localhost:42002", alias="LMS_API_URL")
    lms_api_key: str = Field(default="", alias="LMS_API_KEY")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_api_base_url: str = Field(default="", alias="LLM_API_BASE_URL")
    llm_api_model: str = Field(default="qwen3-coder-flash", alias="LLM_API_MODEL")

    model_config = SettingsConfigDict(
        env_file=".env.bot.secret",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow",
    )


settings = BotSettings.model_validate({})
