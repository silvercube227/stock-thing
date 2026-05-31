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
    supabase_publishable_key: str = Field(default="")  # sb_publishable_... (browser-safe, RLS applies)
    supabase_secret_key: str = Field(default="")       # sb_secret_... (server-only, bypasses RLS)
    database_url: str = Field(default="")

    @property
    def supabase_jwks_url(self) -> str:
        """Public JWKS endpoint for verifying Supabase-issued JWTs."""
        return f"{self.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"

    sec_edgar_user_agent: str = Field(default="stock-thing local dev contact@example.com")

    # LSEG Workspace (analyst estimates). Uses the lseg.data desktop session, which
    # talks to a locally-running Workspace; the app key authorizes that session.
    lseg_app_key: str = Field(default="")

    models_dir: Path = MODELS_DIR
    # On-disk cache of load_frames() pulls, so repeated local experiment/backtest
    # runs read frames from disk instead of re-pulling full history through the
    # Supabase pooler (the dominant Shared-Pooler egress source). Experiments
    # only — production inference always pulls fresh.
    frame_cache_dir: Path = REPO_ROOT / ".frame_cache"
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000)
    # Comma-separated extra CORS origins for the hosted frontend (e.g. the Vercel
    # URL). The localhost dev origins are always allowed; this adds to them.
    cors_origins: str = Field(default="")


@lru_cache
def get_settings() -> Settings:
    return Settings()
