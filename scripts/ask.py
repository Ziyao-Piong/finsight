"""Phase 1 milestone (upgraded in Phase 3): ask a question, get a *cited* answer.

This is the read half of the RAG loop: type a question, watch a grounded answer stream
back, then see the exact passages it was drawn from. Run from the repo root with the venv
active (after `python -m scripts.ingest`):

    python -m scripts.ask
    python -m scripts.ask "What were the main risk factors?"

All retrieve->augment->generate logic lives in src.rag.ask (reused by the API/agent in
later phases); this script just renders the result for a human.
"""

from __future__ import annotations

import sys

from src.rag import ask
from src.retrieval.retriever import Citation

DEFAULT_QUESTION = "What were the main risk factors?"


def format_sources(citations: list[Citation]) -> str:
    """Render citations as a numbered, human-readable Sources block (empty if none).

    The numbers match the ``[i]`` passage labels the model saw, and each line shows the
    chunk id so a reviewer can trace an answer back to an exact, attributed passage.
    """
    if not citations:
        return ""
    lines = ["", "Sources"]
    for i, c in enumerate(citations, start=1):
        lines.append(
            f"[{i}] {c.company} {c.form_type} FY{c.fiscal_year} — {c.section}  ({c.chunk_id})"
        )
    return "\n".join(lines)


def main() -> int:
    # Filings (and answers about them) contain typographic characters like curly quotes;
    # force UTF-8 so they don't garble on Windows' default cp1252 console.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    question = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION

    print(f"Question : {question}")
    print("-" * 60)

    answer = ask(question)  # streams the grounded answer to stdout
    sources = format_sources(answer.citations)
    if sources:
        print(sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
