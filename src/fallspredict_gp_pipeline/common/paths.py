"""Cohort package location and shared path resolution."""
from pathlib import Path
from dare_wearables.common.paths import project_root, resolve_path


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]
