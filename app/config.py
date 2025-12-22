"""
Application configuration using Pydantic Settings.
Loads from environment variables and .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # API Keys
    google_api_key: str = ""
    
    # LangSmith (optional)
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "lawyers-dashboard"
    
    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000
    
    @property
    def project_root(self) -> Path:
        """Project root directory (LawyersDashboard/)."""
        return Path(__file__).parent.parent.parent
    
    @property
    def indexed_files_dir(self) -> Path:
        """
        Directory containing indexed files.
        Defaults to project/.indexedfiles if a project is active,
        otherwise falls back to global LawyersBackend/.indexedfiles.
        """
        # Import here to avoid circular dependency
        from app.middleware.virtual_fs import get_vfs
        
        vfs = get_vfs()
        if vfs._project_path:
            return vfs._project_path / ".indexedfiles"
            
        return self.project_root / ".indexedfiles"
    
    @property
    def langsmith_enabled(self) -> bool:
        """Check if LangSmith tracing is properly configured."""
        return bool(self.langchain_api_key and self.langchain_tracing_v2)
    
    @property
    def templates_dir(self) -> Path:
        """Global templates directory (Documents/LawyersAI/Templates)."""
        return Path.home() / "Documents" / "LawyersAI" / "Templates"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

