"""
Application configuration using Pydantic Settings.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from dotenv import load_dotenv

# Load .env file explicitly
load_dotenv(Path(__file__).parent.parent / ".env")


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Database
    database_url: str = "postgresql+asyncpg://localhost:5432/lawyers_analytics"
    
    # Stripe
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    
    # API Security
    api_key: str = ""
    
    # CORS
    cors_origins: str = "*"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8001


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
