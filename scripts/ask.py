"""Phase 1 milestone: ask a question about the ingested 10-K, get a grounded answer.

This is the *read* half of the walking-skeleton RAG loop and the thing the phase is
really about: type a question, watch a cited-to-the-document answer stream back.

Run it from the repo root with the venv active (after `python -m scripts.ingest`):

    python -m scripts.ask
    python -m scripts.ask "What were the main risk factors?"
    python -m scripts.ask "What products does the company sell?"

It's a thin wrapper over src.rag.ask — all the retrieve->augment->generate logic
lives there so it can be reused by the API/agent in later phases.
"""

from __future__ import annotations

import sys

from src.rag import ask

DEFAULT_QUESTION = "What were the main risk factors?"


def main() -> int:
    # Filings (and the model's answers about them) contain typographic characters
    # like curly quotes; force UTF-8 so they don't garble on Windows' default cp1252
    # console. errors="replace" keeps a stray glyph from crashing the stream.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    question = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION

    print(f"Question : {question}")
    print("-" * 60)

    ask(question)  # streams the grounded answer to stdout
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
