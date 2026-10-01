from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    source_path: str
    start_line: int
    end_line: int
    text: str


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
