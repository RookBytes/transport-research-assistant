from __future__ import annotations

import hashlib
import re
from pathlib import Path

from transport_rag.models import Chunk


def _chunk_id(source_path: str, start_line: int, end_line: int, text: str) -> str:
    payload = f"{source_path}:{start_line}:{end_line}:{text}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def chunk_text(
    text: str,
    source_path: str,
    target_chars: int = 1800,
    overlap_lines: int = 4,
) -> list[Chunk]:
    """
    Structure-aware line chunker.

    Boundaries prefer blank lines and Markdown headings while keeping chunks
    reasonably close to target_chars. Line numbers always refer to the source.
    """
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    start = 0
    n = len(lines)

    while start < n:
        end = start
        chars = 0
        preferred_end = None

        while end < n:
            line = lines[end]
            chars += len(line) + 1

            if end > start and (not line.strip() or re.match(r"^#{1,6}\s", line)):
                preferred_end = end

            if chars >= target_chars:
                break
            end += 1

        if end >= n:
            end = n
        else:
            end = end + 1
            if preferred_end is not None and preferred_end - start >= 5:
                end = preferred_end

        if end <= start:
            end = min(start + 1, n)

        chunk_lines = lines[start:end]
        chunk_body = "\n".join(chunk_lines).strip()

        if chunk_body:
            start_line = start + 1
            end_line = end
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(source_path, start_line, end_line, chunk_body),
                    source_path=source_path,
                    start_line=start_line,
                    end_line=end_line,
                    text=chunk_body,
                )
            )

        if end >= n:
            break

        start = max(end - overlap_lines, start + 1)

    return chunks


def chunk_file(path: Path, root: Path, text: str) -> list[Chunk]:
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    return chunk_text(text=text, source_path=rel)
