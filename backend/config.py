"""Backend configuration loaded from local `.env` secrets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SupabaseConfig:
    url: str | None
    key: str | None
    bucket: str | None
    runs_table: str = "hyperreasoning_runs"

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.key and self.bucket)


@dataclass(frozen=True)
class BackendConfig:
    supabase: SupabaseConfig


def load_backend_config(env_path: Path | None = None) -> BackendConfig:
    """Load backend config from `.env` without exposing secret values."""

    dotenv_path = env_path or ROOT / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)

    return BackendConfig(
        supabase=SupabaseConfig(
            url=_blank_to_none(os.getenv("HYPERREASONING_SUPABASE_URL")),
            key=_blank_to_none(os.getenv("HYPERREASONING_SUPABASE_KEY")),
            bucket=_blank_to_none(os.getenv("HYPERREASONING_SUPABASE_BUCKET")),
            runs_table=os.getenv("HYPERREASONING_SUPABASE_RUNS_TABLE", "hyperreasoning_runs")
            or "hyperreasoning_runs",
        )
    )


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None
