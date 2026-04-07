"""Sync state — tracks file mtimes to detect changes without hashing every file."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


STATE_FILE = "sync_state.json"


@dataclass
class FileSyncRecord:
    """Tracks one file's last-known sync state."""
    path: str           # relative to the root it belongs to
    mtime: float        # os.stat mtime at last sync
    synced_at: str      # ISO timestamp of last successful sync
    direction: str      # "lace_to_obs" | "obs_to_lace" | "initial"
    memory_id: str = "" # mem_xxxx extracted from filename


@dataclass
class SyncState:
    """Full sync state persisted to disk."""
    obsidian_vault: str = ""          # absolute path to Obsidian vault
    last_full_sync: str = ""          # ISO timestamp
    lace_files: dict[str, FileSyncRecord] = field(default_factory=dict)
    obs_files:  dict[str, FileSyncRecord] = field(default_factory=dict)

    # ── Persistence ───────────────────────────────────────────────────────────

    @classmethod
    def load(cls, lace_home: Path) -> "SyncState":
        """Load state from disk. Returns empty state if file doesn't exist."""
        state_file = _state_path(lace_home)
        if not state_file.exists():
            return cls()
        try:
            raw = json.loads(state_file.read_text(encoding="utf-8"))
            state = cls(
                obsidian_vault=raw.get("obsidian_vault", ""),
                last_full_sync=raw.get("last_full_sync", ""),
            )
            for rel, rec in raw.get("lace_files", {}).items():
                state.lace_files[rel] = FileSyncRecord(**rec)
            for rel, rec in raw.get("obs_files", {}).items():
                state.obs_files[rel] = FileSyncRecord(**rec)
            return state
        except Exception:
            return cls()

    def save(self, lace_home: Path) -> None:
        """Persist state to disk."""
        state_file = _state_path(lace_home)
        state_file.parent.mkdir(parents=True, exist_ok=True)

        raw = {
            "obsidian_vault": self.obsidian_vault,
            "last_full_sync": self.last_full_sync,
            "lace_files": {k: asdict(v) for k, v in self.lace_files.items()},
            "obs_files":  {k: asdict(v) for k, v in self.obs_files.items()},
        }
        state_file.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def mark_synced_lace(self, rel: str, mtime: float, direction: str, memory_id: str = "") -> None:
        self.lace_files[rel] = FileSyncRecord(
            path=rel,
            mtime=mtime,
            synced_at=_now_iso(),
            direction=direction,
            memory_id=memory_id,
        )

    def mark_synced_obs(self, rel: str, mtime: float, direction: str, memory_id: str = "") -> None:
        self.obs_files[rel] = FileSyncRecord(
            path=rel,
            mtime=mtime,
            synced_at=_now_iso(),
            direction=direction,
            memory_id=memory_id,
        )

    def lace_file_changed(self, rel: str, current_mtime: float) -> bool:
        """True if this LACE file is new or modified since last sync."""
        rec = self.lace_files.get(rel)
        if rec is None:
            return True
        return current_mtime > rec.mtime + 0.1   # 0.1s tolerance

    def obs_file_changed(self, rel: str, current_mtime: float) -> bool:
        """True if this Obsidian file is new or modified since last sync."""
        rec = self.obs_files.get(rel)
        if rec is None:
            return True
        return current_mtime > rec.mtime + 0.1


def _state_path(lace_home: Path) -> Path:
    return lace_home / "vault" / STATE_FILE


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
