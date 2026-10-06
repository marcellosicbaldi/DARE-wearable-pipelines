from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path | None, *, base: str | Path | None = None) -> Path | None:
    """Resolve against an explicit base, or the caller's working directory."""
    if path is None:
        return None

    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        root = Path(base).expanduser() if base is not None else Path.cwd()
        resolved = root / resolved
    return resolved.resolve()
