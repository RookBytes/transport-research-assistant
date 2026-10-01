from __future__ import annotations

import argparse
from pathlib import Path

from transport_rag.config import settings
from transport_rag.ingestion.chunking import chunk_file
from transport_rag.ingestion.loaders import iter_source_files, read_text
from transport_rag.retrieval.index import DenseIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the transportation RAG index.")
    parser.add_argument("--source", type=Path, required=True, help="Source repository/folder.")
    parser.add_argument("--index", type=Path, default=settings.index_dir, help="Index directory.")
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        raise SystemExit(f"Source does not exist: {source}")

    chunks = []
    file_count = 0

    for path in iter_source_files(source):
        file_count += 1
        chunks.extend(chunk_file(path, source, read_text(path)))

    print(f"Loaded {file_count} source files into {len(chunks)} chunks.")

    index = DenseIndex(settings.embed_model, ollama_url=settings.ollama_url)
    index.build(chunks)
    index.save(args.index.resolve())
    print(f"Saved index to {args.index.resolve()}")


if __name__ == "__main__":
    main()
