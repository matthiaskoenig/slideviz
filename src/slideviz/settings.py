"""Data, index, prediction, mesh and stain reference paths from SLIDEVIZ_* variables or a .env in the working directory.

    export SLIDEVIZ_DATA=/path/to/slides
"""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_cache_dir
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paths the tools need, overridable per machine."""

    model_config = SettingsConfigDict(env_prefix="SLIDEVIZ_", env_file=".env", extra="ignore")

    data: Path | None = None  # set per machine through SLIDEVIZ_DATA, .env or --data
    db: Path = Path(user_cache_dir("slideviz")) / "slides.db"
    # per-slide model output, added as layers when a block that has some is opened
    predictions: Path | None = None
    # mesh/fields.py meshes per animal, added as layers when its block is opened
    meshes: Path | None = None
    # stain reference from stain_stats.py; slides are matched onto its target when found
    stain_reference: Path | None = None

    def predictions_dir(self) -> Path | None:
        """The prediction directory, falling back to one beside the data."""
        if self.predictions is not None:
            return self.predictions
        return self.data / "predictions" if self.data else None

    def meshes_dir(self) -> Path | None:
        """The mesh directory, falling back to one beside the data."""
        if self.meshes is not None:
            return self.meshes
        return self.data / "meshes" if self.data else None

    def stain_reference_file(self) -> Path | None:
        """The stain reference, falling back to the conventional name beside the data."""
        if self.stain_reference is not None:
            return self.stain_reference
        if self.data is None:
            return None
        beside = self.data / "stain_reference.json"
        return beside if beside.exists() else None


settings = Settings()
