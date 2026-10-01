from __future__ import annotations

import argparse

from transport_rag.config import settings
from transport_rag.rag import RAGService


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the transportation research assistant.")
    parser.add_argument("question", help="Question to answer from the indexed sources.")
    args = parser.parse_args()

    service = RAGService(settings)
    result = service.answer(args.question)

    print()
    print(result["answer"])
    if result["sources"]:
        print("\nRetrieved sources:")
        for src in result["sources"]:
            print(
                f"[{src['id']}] {src['source_path']}:{src['start_line']}-{src['end_line']} "
                f"(score={src['score']:.3f})"
            )


if __name__ == "__main__":
    main()
