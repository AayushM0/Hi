"""Tests for vault sync state."""

from __future__ import annotations

import time
from pathlib import Path
import pytest
from lace.vault.state import SyncState, FileSyncRecord, _now_iso


class TestSyncState:
    """Test SyncState persistence and helpers."""

    def test_empty_state_on_missing_file(self, tmp_path):
        """Load returns empty state when no file exists."""
        state = SyncState.load(tmp_path)
        assert state.obsidian_vault == ""
        assert state.last_full_sync == ""
        assert state.lace_files == {}
        assert state.obs_files == {}

    def test_save_and_reload(self, tmp_path):
        """State round-trips through disk correctly."""
        state = SyncState.load(tmp_path)
        state.obsidian_vault = "/home/user/ObsidianVault"
        state.last_full_sync = "2024-01-01T00:00:00Z"
        state.mark_synced_lace("memories/mem_abc123.md", 1234567890.0, "lace_to_obs", "mem_abc123")
        state.save(tmp_path)

        loaded = SyncState.load(tmp_path)
        assert loaded.obsidian_vault == "/home/user/ObsidianVault"
        assert loaded.last_full_sync == "2024-01-01T00:00:00Z"
        assert "memories/mem_abc123.md" in loaded.lace_files

        rec = loaded.lace_files["memories/mem_abc123.md"]
        assert rec.mtime == 1234567890.0
        assert rec.direction == "lace_to_obs"
        assert rec.memory_id == "mem_abc123"

    def test_mark_synced_lace(self, tmp_path):
        """mark_synced_lace creates correct FileSyncRecord."""
        state = SyncState.load(tmp_path)
        state.mark_synced_lace("test.md", 999.0, "obs_to_lace", "mem_xyz")

        rec = state.lace_files["test.md"]
        assert rec.path == "test.md"
        assert rec.mtime == 999.0
        assert rec.direction == "obs_to_lace"
        assert rec.memory_id == "mem_xyz"
        assert rec.synced_at  # not empty

    def test_mark_synced_obs(self, tmp_path):
        """mark_synced_obs creates correct FileSyncRecord."""
        state = SyncState.load(tmp_path)
        state.mark_synced_obs("LACE/test.md", 888.0, "lace_to_obs")

        rec = state.obs_files["LACE/test.md"]
        assert rec.mtime == 888.0
        assert rec.direction == "lace_to_obs"

    def test_lace_file_changed_new_file(self, tmp_path):
        """New file (not in state) is considered changed."""
        state = SyncState.load(tmp_path)
        assert state.lace_file_changed("new_file.md", 1234.0) is True

    def test_lace_file_changed_same_mtime(self, tmp_path):
        """File with same mtime is not considered changed."""
        state = SyncState.load(tmp_path)
        state.mark_synced_lace("test.md", 1000.0, "lace_to_obs")
        assert state.lace_file_changed("test.md", 1000.0) is False

    def test_lace_file_changed_newer_mtime(self, tmp_path):
        """File with newer mtime is considered changed."""
        state = SyncState.load(tmp_path)
        state.mark_synced_lace("test.md", 1000.0, "lace_to_obs")
        assert state.lace_file_changed("test.md", 1001.0) is True

    def test_obs_file_changed_new_file(self, tmp_path):
        """New obs file is considered changed."""
        state = SyncState.load(tmp_path)
        assert state.obs_file_changed("LACE/new.md", 5000.0) is True

    def test_obs_file_changed_same_mtime(self, tmp_path):
        """Obs file with same mtime not changed."""
        state = SyncState.load(tmp_path)
        state.mark_synced_obs("LACE/test.md", 5000.0, "lace_to_obs")
        assert state.obs_file_changed("LACE/test.md", 5000.0) is False

    def test_corrupted_state_returns_empty(self, tmp_path):
        """Corrupted state file returns empty state."""
        state_file = tmp_path / "vault" / "sync_state.json"
        state_file.parent.mkdir(parents=True)
        state_file.write_text("not valid json{{{{")

        state = SyncState.load(tmp_path)
        assert state.lace_files == {}

    def test_now_iso_format(self):
        """_now_iso returns valid ISO string."""
        iso = _now_iso()
        assert "T" in iso
        assert iso.endswith("Z")
        assert len(iso) == 20
