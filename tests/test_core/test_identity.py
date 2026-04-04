"""Tests for identity layer loading."""

import pytest
from pathlib import Path
from lace.core.identity import compose_identity, load_global_identity, load_global_preferences


@pytest.fixture
def lace_home(tmp_path):
    """Create a temporary ~/.lace directory with sample config."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Global identity.md
    identity_file = config_dir / "identity.md"
    identity_file.write_text("# Global Identity\n\nYou are an engineering assistant.")

    # Global preferences.yaml
    prefs_file = config_dir / "preferences.yaml"
    prefs_file.write_text("coding:\n  language: python\n  style: functional")

    return tmp_path


def test_load_global_identity(lace_home):
    identity = load_global_identity(lace_home)
    assert "Global Identity" in identity
    assert "engineering assistant" in identity


def test_load_global_preferences(lace_home):
    prefs = load_global_preferences(lace_home)
    assert prefs["coding"]["language"] == "python"
    assert prefs["coding"]["style"] == "functional"


def test_compose_identity_global(lace_home):
    identity, prefs = compose_identity(lace_home)
    assert "Global Identity" in identity
    assert prefs["coding"]["language"] == "python"


def test_compose_identity_project(lace_home):
    """Project identity and preferences override global."""
    project_dir = lace_home / "config" / "projects" / "my-api"
    project_dir.mkdir(parents=True)

    # Project identity.md
    project_identity = project_dir / "identity.md"
    project_identity.write_text("# Project Identity\n\nThis is a FastAPI project.")

    # Project preferences.yaml
    project_prefs = project_dir / "preferences.yaml"
    project_prefs.write_text("coding:\n  style: object-oriented\n  testing_framework: pytest")

    identity, prefs = compose_identity(lace_home, scope="project:my-api")
    assert "Global Identity" in identity
    assert "Project Identity" in identity
    assert "FastAPI project" in identity
    assert prefs["coding"]["style"] == "object-oriented"  # project overrides global
    assert prefs["coding"]["language"] == "python"        # global still present
    assert prefs["coding"]["testing_framework"] == "pytest"  # project addition