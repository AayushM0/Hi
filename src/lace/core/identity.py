"""Identity layer loading — compose identity.md + preferences.yaml.

This module loads:
  1. Global identity.md and preferences.yaml from ~/.lace/config/
  2. Project-specific overlays from ~/.lace/config/projects/<name>/
  3. Returns a composed identity string for use in prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lace.core.config import LaceConfig


def load_global_identity(lace_home: Path) -> str:
    """Load global identity.md from ~/.lace/config/."""
    identity_file = lace_home / "config" / "identity.md"
    if not identity_file.exists():
        return "You are an engineering assistant working with a senior developer."

    try:
        return identity_file.read_text(encoding="utf-8")
    except Exception:
        return "You are an engineering assistant working with a senior developer."


def load_global_preferences(lace_home: Path) -> dict:
    """Load global preferences.yaml from ~/.lace/config/."""
    preferences_file = lace_home / "config" / "preferences.yaml"
    if not preferences_file.exists():
        return {}

    try:
        import yaml
        with open(preferences_file) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_project_identity(lace_home: Path, project_name: str) -> str:
    """Load project-specific identity.md from ~/.lace/config/projects/<name>/."""
    project_dir = lace_home / "config" / "projects" / project_name
    identity_file = project_dir / "identity.md"
    if not identity_file.exists():
        return ""

    try:
        return identity_file.read_text(encoding="utf-8")
    except Exception:
        return ""


def load_project_preferences(lace_home: Path, project_name: str) -> dict:
    """Load project-specific preferences.yaml from ~/.lace/config/projects/<name>/."""
    project_dir = lace_home / "config" / "projects" / project_name
    preferences_file = project_dir / "preferences.yaml"
    if not preferences_file.exists():
        return {}

    try:
        import yaml
        with open(preferences_file) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def compose_identity(
    lace_home: Path,
    scope: str = "global",
    config: LaceConfig | None = None,
) -> tuple[str, dict]:
    """Compose full identity string and preferences for current scope.

    Global identity + project identity (if any) → combined string.
    Global preferences + project preferences (if any) → merged dict.

    Returns:
        (identity_string, merged_preferences)
    """
    # Load global identity and preferences
    global_identity = load_global_identity(lace_home)
    global_prefs = load_global_preferences(lace_home)

    # Load project-specific overlays if scope is project
    project_identity = ""
    project_prefs = {}
    if scope.startswith("project:"):
        project_name = scope.removeprefix("project:")
        project_identity = load_project_identity(lace_home, project_name)
        project_prefs = load_project_preferences(lace_home, project_name)

    # Merge identity strings
    full_identity = global_identity
    if project_identity:
        full_identity += "\n\n--- Project Identity ---\n\n" + project_identity

    # Merge preferences (project overrides global, deeply)
    merged_prefs = _deep_merge(global_prefs, project_prefs)

    return full_identity, merged_prefs