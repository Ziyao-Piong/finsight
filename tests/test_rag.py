"""Tests for src/rag.py ask() wiring (Phase 3).

ask() is glue: retrieve -> build context -> generate -> attach citations. We stub both
the retriever and the chat model so the test is deterministic and offline, and assert
the wiring: citations are carried onto the Answer, and the streamed text is returned.
"""

from __future__ import annotations

import src.rag as rag
from src.rag import Answer, ask
from src.retrieval.retriever import Citation


class _FakeChunk:
    def __init__(self, content):
        self.content = content


class _FakeChat:
    def stream(self, messages):
        for tok in ["Revenue ", "grew ", "24%."]:
            yield _FakeChunk(tok)


def _citation(ticker, year, section):
    return Citation(
        company="NVIDIA Corporation",
        ticker=ticker,
        form_type="10-K",
        fiscal_year=year,
        section=section,
        chunk_id=f"{ticker}-{year}-10-k-{section}-1",
        char_start=0,
        char_end=10,
        source_url="https://sec.gov/x.htm",
        snippet="net revenue increased",
        score=0.1,
    )


def test_ask_returns_answer_with_citations(monkeypatch):
    cites = [_citation("NVDA", 2024, "MD&A")]
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: cites)
    monkeypatch.setattr(rag, "get_chat_model", lambda settings=None: _FakeChat())

    answer = ask("How did revenue change?", stream=False)

    assert isinstance(answer, Answer)
    assert answer.text == "Revenue grew 24%."
    assert answer.citations == cites


def test_ask_with_no_results_returns_empty_citations(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: [])
    # chat model must NOT be called when there are no passages
    monkeypatch.setattr(
        rag, "get_chat_model", lambda settings=None: (_ for _ in ()).throw(AssertionError("called"))
    )

    answer = ask("Anything?", stream=False)

    assert isinstance(answer, Answer)
    assert answer.citations == []
    assert "ingest" in answer.text.lower()


def test_format_sources_lists_numbered_citations():
    from scripts.ask import format_sources

    out = format_sources([_citation("NVDA", 2024, "MD&A"), _citation("NVDA", 2023, "Risk Factors")])
    assert "Sources" in out
    assert "[1] NVIDIA Corporation 10-K FY2024 — MD&A" in out
    assert "[2] NVIDIA Corporation 10-K FY2023 — Risk Factors" in out
    assert "NVDA-2024-10-k-MD&A-1" in out  # chunk id shown for verifiability


def test_format_sources_empty_is_blank():
    from scripts.ask import format_sources

    assert format_sources([]) == ""
