"""Core sync logic between LACE vault and Obsidian vault."""

from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from lace.vault.state import SyncState, _now_iso


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class SyncResult:
    """Summary of what happened during a sync operation."""
    lace_to_obs: list[str] = field(default_factory=list)
    obs_to_lace: list[str] = field(default_factory=list)
    reindexed:   list[str] = field(default_factory=list)
    skipped:     list[str] = field(default_factory=list)
    errors:      list[str] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        return len(self.lace_to_obs) + len(self.obs_to_lace)


# ── Memory ID extraction ──────────────────────────────────────────────────────

_MEM_ID_RE = re.compile(r"mem_[0-9a-f]{12}")

def _extract_memory_id(path: Path) -> str:
    m = _MEM_ID_RE.search(path.stem)
    return m.group(0) if m else ""


def _is_memory_file(path: Path) -> bool:
    return path.suffix == ".md" and bool(_MEM_ID_RE.search(path.stem))


# ── Obsidian subfolder mirror path ───────────────────────────────────────────

def _obs_mirror_path(lace_file: Path, lace_vault: Path, obs_vault: Path) -> Path:
    rel = lace_file.relative_to(lace_vault)
    return obs_vault / "LACE" / rel


def _lace_source_path(obs_file: Path, obs_vault: Path, lace_vault: Path) -> Path | None:
    try:
        rel = obs_file.relative_to(obs_vault / "LACE")
        return lace_vault / rel
    except ValueError:
        return None


# ── Main sync functions ───────────────────────────────────────────────────────

def full_sync(
    lace_vault: Path,
    obs_vault: Path,
    lace_home: Path,
    reindex: bool = True,
) -> SyncResult:
    """Full bidirectional sync between LACE vault and Obsidian vault."""
    result = SyncResult()
    state = SyncState.load(lace_home)
    state.obsidian_vault = str(obs_vault)

    for lace_file in sorted(lace_vault.rglob("*.md")):
        if not _is_memory_file(lace_file):
            continue

        rel = str(lace_file.relative_to(lace_vault))
        lace_mtime = lace_file.stat().st_mtime
        obs_file = _obs_mirror_path(lace_file, lace_vault, obs_vault)

        if not obs_file.exists():
            _copy_file(lace_file, obs_file, result, "lace_to_obs")
            state.mark_synced_lace(rel, lace_mtime, "lace_to_obs", _extract_memory_id(lace_file))
            obs_rel = str(obs_file.relative_to(obs_vault))
            state.mark_synced_obs(obs_rel, obs_file.stat().st_mtime, "lace_to_obs")
        else:
            obs_mtime = obs_file.stat().st_mtime

            if lace_mtime > obs_mtime + 1.0:
                _copy_file(lace_file, obs_file, result, "lace_to_obs")
                state.mark_synced_lace(rel, lace_mtime, "lace_to_obs", _extract_memory_id(lace_file))
                obs_rel = str(obs_file.relative_to(obs_vault))
                state.mark_synced_obs(obs_rel, obs_file.stat().st_mtime, "lace_to_obs")

            elif obs_mtime > lace_mtime + 1.0:
                mem_id = _pull_obs_to_lace(obs_file, lace_file, lace_vault, lace_home, result, reindex)
                state.mark_synced_lace(rel, lace_file.stat().st_mtime, "obs_to_lace", mem_id)
                obs_rel = str(obs_file.relative_to(obs_vault))
                state.mark_synced_obs(obs_rel, obs_mtime, "obs_to_lace", mem_id)
            else:
                result.skipped.append(rel)

    obs_lace_dir = obs_vault / "LACE"
    if obs_lace_dir.exists():
        for obs_file in sorted(obs_lace_dir.rglob("*.md")):
            if not _is_memory_file(obs_file):
                continue

            lace_file = _lace_source_path(obs_file, obs_vault, lace_vault)
            if lace_file is None:
                continue

            if not lace_file.exists():
                mem_id = _pull_obs_to_lace(obs_file, lace_file, lace_vault, lace_home, result, reindex)
                rel = str(lace_file.relative_to(lace_vault))
                state.mark_synced_lace(rel, lace_file.stat().st_mtime, "obs_to_lace", mem_id)
                obs_rel = str(obs_file.relative_to(obs_vault))
                state.mark_synced_obs(obs_rel, obs_file.stat().st_mtime, "obs_to_lace", mem_id)

    state.last_full_sync = _now_iso()
    state.save(lace_home)
    return result


def _copy_file(src: Path, dst: Path, result: SyncResult, direction: str) -> None:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if direction == "lace_to_obs":
            result.lace_to_obs.append(src.name)
        else:
            result.obs_to_lace.append(src.name)
    except Exception as e:
        result.errors.append(f"{src.name}: {e}")


def _pull_obs_to_lace(
    obs_file: Path,
    lace_file: Path,
    lace_vault: Path,
    lace_home: Path,
    result: SyncResult,
    reindex: bool,
) -> str:
    mem_id = _extract_memory_id(obs_file)
    try:
        lace_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(obs_file, lace_file)
        result.obs_to_lace.append(obs_file.name)

        if reindex and mem_id:
            _reindex_file(lace_file, lace_home)
            result.reindexed.append(mem_id)
    except Exception as e:
        result.errors.append(f"{obs_file.name}: {e}")
        return ""
    return mem_id


def _reindex_file(md_file: Path, lace_home: Path) -> None:
    from lace.memory.markdown import markdown_to_memory
    from lace.retrieval.embeddings import embed_text
    from lace.retrieval.vector import upsert_memory
    from lace.core.config import load_config

    config = load_config(lace_home)
    memory = markdown_to_memory(md_file)
    if memory is None:
        return

    memory.embedding = embed_text(memory.content, model_name=config.embeddings.model)
    vector_db_path = lace_home / "memory" / "vector_db"
    upsert_memory(memory, vector_db_path)


# ── Single-file sync (used by watcher) ───────────────────────────────────────

_last_synced: dict[str, float] = {}

def sync_single_file(
    changed_file: Path,
    lace_vault: Path,
    obs_vault: Path,
    lace_home: Path,
) -> SyncResult:
    """Sync a single changed file. Called by the file watcher."""
    result = SyncResult()
    
    file_key = str(changed_file)
    now = time.time()
    if now - _last_synced.get(file_key, 0) < 2.0:
        return result
    
    _last_synced[file_key] = now

    try:
        changed_file.relative_to(lace_vault)
        in_lace = True
    except ValueError:
        in_lace = False

    try:
        changed_file.relative_to(obs_vault / "LACE")
        in_obs = True
    except ValueError:
        in_obs = False

    if in_lace:
        obs_file = _obs_mirror_path(changed_file, lace_vault, obs_vault)
        
        # Check if content is already identical before copying
        if obs_file.exists() and _files_identical(changed_file, obs_file):
            result.skipped.append(changed_file.name)
            return result
            
        _copy_file(changed_file, obs_file, result, "lace_to_obs")
        _last_synced[str(obs_file)] = now

    elif in_obs:
        lace_file = _lace_source_path(changed_file, obs_vault, lace_vault)
        if lace_file:
            # Check if content is already identical
            if lace_file.exists() and _files_identical(changed_file, lace_file):
                result.skipped.append(changed_file.name)
                return result
                
            _pull_obs_to_lace(changed_file, lace_file, lace_vault, lace_home, result, reindex=True)
            _last_synced[str(lace_file)] = now

    return result


def _files_identical(path1: Path, path2: Path) -> bool:
    """Check if two files have identical content."""
    try:
        return path1.read_bytes() == path2.read_bytes()
    except Exception:
        return False


# ── Status helpers ────────────────────────────────────────────────────────────

def get_sync_status(lace_home: Path) -> dict:
    state = SyncState.load(lace_home)
    return {
        "obsidian_vault":  state.obsidian_vault or None,
        "last_full_sync":  state.last_full_sync or None,
        "lace_files_tracked": len(state.lace_files),
        "obs_files_tracked":  len(state.obs_files),
        "configured": bool(state.obsidian_vault),
    }
