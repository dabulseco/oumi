from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EXCLUDES = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "venv",
    "node_modules",
}

BINARY_MIME_PREFIXES = ("image/", "audio/", "video/", "application/octet-stream")


@dataclass(slots=True)
class RepoFile:
    path: Path
    rel_path: str
    suffix: str


def _is_probably_binary(path: Path) -> bool:
    mime, _ = mimetypes.guess_type(path.name)
    return bool(mime and any(mime.startswith(prefix) for prefix in BINARY_MIME_PREFIXES))


def scan_repository(repo_root: Path, max_file_size_bytes: int) -> list[RepoFile]:
    """Scan repository files, excluding known noisy dirs and likely binaries."""
    files: list[RepoFile] = []

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(repo_root)
        parts = set(rel.parts)
        if parts & DEFAULT_EXCLUDES:
            continue
        if path.stat().st_size > max_file_size_bytes:
            continue
        if _is_probably_binary(path):
            continue

        files.append(RepoFile(path=path, rel_path=str(rel), suffix=path.suffix.lower()))

    return files
