from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    supabase_url: str = Field(default="")
    supabase_anon_key: str = Field(default="")
    supabase_service_key: str = Field(default="")
    supabase_jwt_secret: str = Field(default="")
    database_url: str = Field(default="")

    sec_edgar_user_agent: str = Field(default="stock-thing local dev contact@example.com")

    models_dir: Path = MODELS_DIR
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000)


@lru_cache
def get_settings() -> Settings:
    return Settings()
