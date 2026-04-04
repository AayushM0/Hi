"""Tests for LACE configuration system."""

import pytest
from pathlib import Path
import yaml

from lace.core.config import (
    LaceConfig,
    MemoryConfig,
    RetrievalConfig,
    init_lace_home,
    load_config,
    save_config,
    set_config_value,
    get_lace_home,
)


def test_default_config_is_valid():
    """LaceConfig initializes with correct defaults."""
    config = LaceConfig()
    assert config.version == "1.0"
    assert config.memory.decay_half_life_days == 30
    assert config.retrieval.relevance_threshold == 0.35
    assert config.retrieval.max_results == 20
    assert config.embeddings.provider == "local"


def test_config_weights_sum_to_one():
    """Retrieval weights should sum to 1.0."""
    config = LaceConfig()
    weights = config.retrieval.weights
    total = (
        weights.semantic_similarity
        + weights.recency
        + weights.frequency
        + weights.confidence
        + weights.scope
    )
    assert abs(total - 1.0) < 0.001


def test_init_creates_directory_structure(tmp_path):
    """lace init creates all required directories."""
    lace_home = tmp_path / ".lace"
    path, already_existed = init_lace_home(lace_home)

    assert not already_existed
    assert (lace_home / "memory" / "vault" / "global" / "patterns").exists()
    assert (lace_home / "memory" / "vault" / "global" / "decisions").exists()
    assert (lace_home / "memory" / "vault" / "global" / "debug-log").exists()
    assert (lace_home / "memory" / "vault" / "global" / "references").exists()
    assert (lace_home / "memory" / "vault" / "projects").exists()
    assert (lace_home / "logs" / "retrieval").exists()
    assert (lace_home / "logs" / "interactions").exists()
    assert (lace_home / "sessions").exists()
    assert (lace_home / "config" / "projects").exists()


def test_init_is_idempotent(tmp_path):
    """Running init twice doesn't fail or overwrite existing files."""
    lace_home = tmp_path / ".lace"

    _, first = init_lace_home(lace_home)
    assert not first

    _, second = init_lace_home(lace_home)
    assert second  # already existed on second run


def test_load_config_missing_file_returns_defaults(tmp_path):
    """Loading config when file doesn't exist returns defaults."""
    lace_home = tmp_path / ".lace"
    lace_home.mkdir()

    config = load_config(lace_home)
    assert isinstance(config, LaceConfig)
    assert config.memory.dedup_threshold == 0.85


def test_save_and_load_roundtrip(tmp_path):
    """Config can be saved and loaded back identically."""
    lace_home = tmp_path / ".lace"
    lace_home.mkdir()
    (lace_home / "config").mkdir()

    config = LaceConfig()
    config.memory.decay_half_life_days = 60
    config.retrieval.max_results = 15

    save_config(config, lace_home)
    loaded = load_config(lace_home)

    assert loaded.memory.decay_half_life_days == 60
    assert loaded.retrieval.max_results == 15


def test_set_config_value_int(tmp_path):
    """set_config_value correctly sets an integer value."""
    lace_home = tmp_path / ".lace"
    init_lace_home(lace_home)

    set_config_value("memory.decay_half_life_days", "60", lace_home)
    config = load_config(lace_home)
    assert config.memory.decay_half_life_days == 60


def test_set_config_value_bool(tmp_path):
    """set_config_value correctly sets a boolean value."""
    lace_home = tmp_path / ".lace"
    init_lace_home(lace_home)

    set_config_value("memory.auto_extract", "true", lace_home)
    config = load_config(lace_home)
    assert config.memory.auto_extract is True


def test_set_config_value_unknown_key_raises(tmp_path):
    """set_config_value raises KeyError for unknown keys."""
    lace_home = tmp_path / ".lace"
    init_lace_home(lace_home)

    with pytest.raises(KeyError):
        set_config_value("memory.nonexistent_key", "value", lace_home)


def test_vault_path_default(tmp_path):
    """vault_path returns default when not configured."""
    lace_home = tmp_path / ".lace"
    config = LaceConfig()
    assert config.vault_path(lace_home) == lace_home / "memory" / "vault"


def test_vault_path_custom():
    """vault_path respects custom path when set."""
    config = LaceConfig()
    config.vault.path = "/custom/vault"
    assert config.vault_path(Path("/anything")) == Path("/custom/vault")