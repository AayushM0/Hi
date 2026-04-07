"""File watcher — monitors LACE vault and Obsidian vault for changes."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileModifiedEvent, FileCreatedEvent, FileSystemEventHandler
from watchdog.observers import Observer


# ── Event handler ─────────────────────────────────────────────────────────────

class VaultEventHandler(FileSystemEventHandler):
    """Handles file system events from watchdog."""

    def __init__(
        self,
        lace_vault: Path,
        obs_vault: Path,
        lace_home: Path,
        on_change: Callable[[Path, str], None],
    ) -> None:
        super().__init__()
        self.lace_vault = lace_vault
        self.obs_vault  = obs_vault
        self.lace_home  = lace_home
        self.on_change  = on_change
        self._last_sync: dict[str, tuple[float, str]] = {}  # path -> (timestamp, direction)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Called when a file is modified."""
        if event.is_directory:
            return
        self._handle_change(Path(str(event.src_path)))

    def on_created(self, event: FileSystemEvent) -> None:
        """Called when a file is created."""
        if event.is_directory:
            return
        self._handle_change(Path(str(event.src_path)))

    def _handle_change(self, src: Path) -> None:
        """Process a file change event."""
        # Only care about .md files
        if src.suffix != ".md":
            return

        # Determine which vault this file belongs to and its mirror
        direction = None
        mirror_path = None
        
        try:
            rel = src.relative_to(self.lace_vault)
            direction = "lace_to_obs"
            mirror_path = self.obs_vault / "LACE" / rel
        except ValueError:
            try:
                rel = src.relative_to(self.obs_vault / "LACE")
                direction = "obs_to_lace"
                mirror_path = self.lace_vault / rel
            except ValueError:
                return  # Not in either vault

        now = time.time()
        src_key = str(src)
        mirror_key = str(mirror_path) if mirror_path else None

        # Check if this change is actually us syncing the mirror
        if mirror_key and mirror_key in self._last_sync:
            last_time, last_dir = self._last_sync[mirror_key]
            if now - last_time < 2.0 and last_dir != direction:
                return  # This is an echo from our own sync

        # Debounce rapid events on same file
        if src_key in self._last_sync:
            last_time, _ = self._last_sync[src_key]
            if now - last_time < 1.0:
                return

        # Record BOTH source and mirror as "just synced" to prevent loops
        self._last_sync[src_key] = (now, direction)
        if mirror_key:
            mirror_direction = "obs_to_lace" if direction == "lace_to_obs" else "lace_to_obs"
            self._last_sync[mirror_key] = (now, mirror_direction)
        
        # Trigger the sync
        self.on_change(src, direction)


# ── Watcher daemon ────────────────────────────────────────────────────────────

def start_watcher(
    lace_vault: Path,
    obs_vault: Path,
    lace_home: Path,
    on_change: Callable[[Path, str], None],
    poll_interval: float = 1.0,
) -> None:
    """Start watchdog observers on both vaults. Blocks until Ctrl+C."""
    from watchdog.observers.polling import PollingObserver

    handler = VaultEventHandler(
        lace_vault=lace_vault,
        obs_vault=obs_vault,
        lace_home=lace_home,
        on_change=on_change,
    )

    observer = PollingObserver(timeout=poll_interval)
    observer.schedule(handler, str(lace_vault), recursive=True)

    obs_lace_dir = obs_vault / "LACE"
    obs_lace_dir.mkdir(parents=True, exist_ok=True)
    observer.schedule(handler, str(obs_lace_dir), recursive=True)

    observer.start()

    try:
        while observer.is_alive():
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
