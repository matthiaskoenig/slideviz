"""Where the data and the index live - from the environment.

    export SLIDEVIZ_DATA=/path/to/slides

Reads a .env file in the working directory too, so a machine can be
configured once without exporting anything.
"""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_cache_dir
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paths the tools need, overridable per machine."""

    model_config = SettingsConfigDict(env_prefix="SLIDEVIZ_", env_file=".env", extra="ignore")

    data: Path | None = None  # no default: a path baked in here only works on one machine
    db: Path = Path(user_cache_dir("slideviz")) / "slides.db"


settings = Settings()
