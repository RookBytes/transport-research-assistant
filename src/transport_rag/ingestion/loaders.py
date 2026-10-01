from __future__ import annotations

from pathlib import Path

SUPPORTED_SUFFIXES = {
    ".md",
    ".txt",
    ".py",
    ".java",
    ".properties",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}

SKIP_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "__pycache__",
    "target",
    "out",
    "build",
    "dist",
    ".rag_index",
}

MAX_FILE_BYTES = 2_000_000


def iter_source_files(root: Path):
    root = root.resolve()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        yield path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")
