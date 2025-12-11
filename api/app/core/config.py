"""Application configuration."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Leap Trading System API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    # CORS settings
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # Paths (relative to project root)
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent
    SAVED_MODELS_DIR: Path = PROJECT_ROOT / "saved_models"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    CONFIG_DIR: Path = PROJECT_ROOT / "config"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
