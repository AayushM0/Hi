"""Configuration management for LACE."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ── Path resolution ──────────────────────────────────────────────────────────

def get_lace_home() -> Path:
    """Return the LACE home directory.
    
    Checks LACE_HOME env var first, falls back to ~/.lace
    """
    env_home = os.environ.get("LACE_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    return Path.home() / ".lace"


def get_config_dir() -> Path:
    """Return the default config templates directory (inside the package)."""
    return Path(__file__).parent.parent.parent.parent / "config"


# ── Config models ─────────────────────────────────────────────────────────────

class MemoryConfig(BaseModel):
    auto_extract: bool = False
    extraction_threshold: float = 0.6
    require_confirmation: bool = False
    max_extractions_per_turn: int = 3
    dedup_threshold: float = 0.85
    decay_half_life_days: int = 30
    consolidation_schedule: str = "weekly"


class RetrievalWeights(BaseModel):
    semantic_similarity: float = 0.40
    recency: float = 0.20
    frequency: float = 0.15
    confidence: float = 0.15
    scope: float = 0.10


class RetrievalConfig(BaseModel):
    relevance_threshold: float = 0.35
    max_results: int = 20
    weights: RetrievalWeights = Field(default_factory=RetrievalWeights)


class VaultConfig(BaseModel):
    obsidian_compatible: bool = True
    file_watcher: bool = False
    path: str | None = None  # None = use default ~/.lace/memory/vault


class LoggingConfig(BaseModel):
    retrieval_logs: bool = True
    interaction_logs: bool = True
    log_retention_days: int = 90


class EmbeddingsConfig(BaseModel):
    provider: str = "local"              # "local" or "openai"
    model: str = "all-MiniLM-L6-v2"     # local default


class LaceConfig(BaseModel):
    """Root configuration model for LACE."""
    version: str = "1.0"
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    vault: VaultConfig = Field(default_factory=VaultConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)

    def vault_path(self, lace_home: Path) -> Path:
        """Resolve the vault path."""
        if self.vault.path:
            return Path(self.vault.path).expanduser().resolve()
        return lace_home / "memory" / "vault"


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(lace_home: Path | None = None) -> LaceConfig:
    """Load configuration from ~/.lace/config/lace.yaml.
    
    Falls back to defaults if file doesn't exist.
    """
    if lace_home is None:
        lace_home = get_lace_home()

    config_file = lace_home / "config" / "lace.yaml"

    if not config_file.exists():
        return LaceConfig()

    with open(config_file) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return LaceConfig.model_validate(raw)


def save_config(config: LaceConfig, lace_home: Path | None = None) -> None:
    """Write config back to lace.yaml."""
    if lace_home is None:
        lace_home = get_lace_home()

    config_file = lace_home / "config" / "lace.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)


def set_config_value(key_path: str, value: str, lace_home: Path | None = None) -> None:
    """Set a nested config value using dot notation.
    
    Example: set_config_value("memory.decay_half_life_days", "60")
    """
    if lace_home is None:
        lace_home = get_lace_home()

    config = load_config(lace_home)
    data = config.model_dump()

    keys = key_path.split(".")
    target = data
    for key in keys[:-1]:
        if key not in target:
            raise KeyError(f"Unknown config key: {key_path}")
        target = target[key]

    final_key = keys[-1]
    if final_key not in target:
        raise KeyError(f"Unknown config key: {key_path}")

    # Type coercion — preserve the original type
    original = target[final_key]
    if isinstance(original, bool):
        target[final_key] = value.lower() in ("true", "1", "yes")
    elif isinstance(original, int):
        target[final_key] = int(value)
    elif isinstance(original, float):
        target[final_key] = float(value)
    else:
        target[final_key] = value

    updated = LaceConfig.model_validate(data)
    save_config(updated, lace_home)


# ── Init system ───────────────────────────────────────────────────────────────

VAULT_SUBDIRS = [
    "global/patterns",
    "global/decisions",
    "global/debug-log",
    "global/references",
    "projects",
]

OTHER_DIRS = [
    "logs/retrieval",
    "logs/interactions",
    "sessions",
]


def init_lace_home(lace_home: Path | None = None) -> tuple[Path, bool]:
    """Create the ~/.lace directory structure.
    
    Returns (lace_home_path, was_already_initialized).
    """
    if lace_home is None:
        lace_home = get_lace_home()

    already_existed = lace_home.exists()

    # Create all directories
    for subdir in VAULT_SUBDIRS:
        (lace_home / "memory" / "vault" / subdir).mkdir(parents=True, exist_ok=True)

    for subdir in OTHER_DIRS:
        (lace_home / subdir).mkdir(parents=True, exist_ok=True)

    (lace_home / "config" / "projects").mkdir(parents=True, exist_ok=True)

    # Copy default config templates (only if they don't already exist)
    templates_dir = get_config_dir()
    config_dest = lace_home / "config"

    for template_file in ["lace.yaml", "identity.md", "preferences.yaml"]:
        src = templates_dir / template_file
        dst = config_dest / template_file
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    return lace_home, already_existed