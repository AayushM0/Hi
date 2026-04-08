"""Tests for vault sync logic."""

from __future__ import annotations

import shutil
import time
import os
from pathlib import Path
import pytest
from lace.vault.sync import (
    full_sync,
    sync_single_file,
    get_sync_status,
    _extract_memory_id,
    _is_memory_file,
    _obs_mirror_path,
    _lace_source_path,
    _files_identical,
)


# fixtures

@pytest.fixture
def lace_vault(tmp_path):
    vault = tmp_path / "lace_vault"
    vault.mkdir()
    return vault

@pytest.fixture
def obs_vault(tmp_path):
    vault = tmp_path / "obsidian_vault"
    vault.mkdir()
    (vault / "LACE").mkdir()
    return vault

@pytest.fixture
def lace_home(tmp_path):
    home = tmp_path / ".lace"
    home.mkdir()
    return home


def make_memory_file(vault: Path, name: str = "mem_abc123def456.md") -> Path:
    f = vault / name
    lines = [
        "---",
        "id: mem_abc123def456",
        "tags: [test]",
        "---",
        "",
        "Test memory content.",
        "",
    ]
    f.write_text("\n".join(lines), encoding="utf-8")
    return f


class TestHelpers:

    def test_extract_memory_id_valid(self):
        p = Path("some_topic_mem_abc123def456.md")
        assert _extract_memory_id(p) == "mem_abc123def456"

    def test_extract_memory_id_missing(self):
        p = Path("identity.md")
        assert _extract_memory_id(p) == ""

    def test_is_memory_file_true(self):
        assert _is_memory_file(Path("topic_mem_abc123def456.md")) is True

    def test_is_memory_file_false_no_id(self):
        assert _is_memory_file(Path("identity.md")) is False

    def test_is_memory_file_false_wrong_ext(self):
        assert _is_memory_file(Path("mem_abc123def456.txt")) is False

    def test_obs_mirror_path(self, tmp_path):
        lace_vault = tmp_path / "lace"
        obs_vault = tmp_path / "obs"
        lace_file = lace_vault / "mem_abc123def456.md"
        mirror = _obs_mirror_path(lace_file, lace_vault, obs_vault)
        assert mirror == obs_vault / "LACE" / "mem_abc123def456.md"

    def test_lace_source_path_valid(self, tmp_path):
        lace_vault = tmp_path / "lace"
        obs_vault = tmp_path / "obs"
        obs_file = obs_vault / "LACE" / "mem_abc123def456.md"
        source = _lace_source_path(obs_file, obs_vault, lace_vault)
        assert source == lace_vault / "mem_abc123def456.md"

    def test_lace_source_path_not_in_lace_dir(self, tmp_path):
        lace_vault = tmp_path / "lace"
        obs_vault = tmp_path / "obs"
        obs_file = obs_vault / "OtherFolder" / "note.md"
        source = _lace_source_path(obs_file, obs_vault, lace_vault)
        assert source is None

    def test_files_identical_same_content(self, tmp_path):
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_bytes(b"hello world")
        b.write_bytes(b"hello world")
        assert _files_identical(a, b) is True

    def test_files_identical_different_content(self, tmp_path):
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_bytes(b"hello")
        b.write_bytes(b"world")
        assert _files_identical(a, b) is False

    def test_files_identical_missing_file(self, tmp_path):
        a = tmp_path / "a.md"
        b = tmp_path / "missing.md"
        a.write_bytes(b"hello")
        assert _files_identical(a, b) is False


class TestFullSync:

    def test_lace_to_obs_new_file(self, lace_vault, obs_vault, lace_home):
        make_memory_file(lace_vault)
        result = full_sync(lace_vault, obs_vault, lace_home, reindex=False)
        mirror = obs_vault / "LACE" / "mem_abc123def456.md"
        assert mirror.exists()
        assert "mem_abc123def456.md" in result.lace_to_obs

    def test_obs_to_lace_newer_file(self, lace_vault, obs_vault, lace_home):
        lace_file = make_memory_file(lace_vault)
        obs_lace = obs_vault / "LACE"
        obs_file = obs_lace / "mem_abc123def456.md"
        lines = [
            "---",
            "id: mem_abc123def456",
            "tags: [updated]",
            "---",
            "",
            "Updated content.",
            "",
        ]
        obs_file.write_text("\n".join(lines), encoding="utf-8")
        old_time = time.time() - 10
        os.utime(lace_file, (old_time, old_time))
        result = full_sync(lace_vault, obs_vault, lace_home, reindex=False)
        assert "mem_abc123def456.md" in result.obs_to_lace

    def test_identical_files_skipped(self, lace_vault, obs_vault, lace_home):
        lace_file = make_memory_file(lace_vault)
        obs_file = obs_vault / "LACE" / "mem_abc123def456.md"
        shutil.copy2(lace_file, obs_file)
        result = full_sync(lace_vault, obs_vault, lace_home, reindex=False)
        assert "mem_abc123def456.md" in result.skipped

    def test_no_memory_files_empty_result(self, lace_vault, obs_vault, lace_home):
        result = full_sync(lace_vault, obs_vault, lace_home, reindex=False)
        assert result.total_changes == 0
        assert result.errors == []


class TestSyncStatus:

    def test_status_unconfigured(self, lace_home):
        status = get_sync_status(lace_home)
        assert status["configured"] is False
        assert status["obsidian_vault"] is None

    def test_status_after_sync(self, lace_vault, obs_vault, lace_home):
        make_memory_file(lace_vault)
        full_sync(lace_vault, obs_vault, lace_home, reindex=False)
        status = get_sync_status(lace_home)
        assert status["obsidian_vault"] == str(obs_vault)
        assert status["configured"] is True
        assert status["last_full_sync"] is not None


class TestSyncSingleFile:

    def test_lace_file_synced_to_obs(self, lace_vault, obs_vault, lace_home):
        lace_file = make_memory_file(lace_vault)
        result = sync_single_file(lace_file, lace_vault, obs_vault, lace_home)
        mirror = obs_vault / "LACE" / "mem_abc123def456.md"
        assert mirror.exists()
        assert "mem_abc123def456.md" in result.lace_to_obs

    def test_identical_file_skipped(self, lace_vault, obs_vault, lace_home):
        lace_file = make_memory_file(lace_vault)
        obs_file = obs_vault / "LACE" / "mem_abc123def456.md"
        shutil.copy2(lace_file, obs_file)
        result = sync_single_file(lace_file, lace_vault, obs_vault, lace_home)
        assert "mem_abc123def456.md" in result.skipped
