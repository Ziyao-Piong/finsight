"""Tests for src/retrieval/retriever.py — filtering + citations (Phase 3, TDD).

Split in three, mirroring conftest's offline philosophy (no network, no API keys):
  * _build_where: a pure function, unit-tested directly.
  * _document_to_citation: a pure mapping, unit-tested directly.
  * retrieve: an integration test over a tiny Chroma built with deterministic fake
    embeddings, proving metadata filtering actually restricts the results.
"""

from __future__ import annotations

import hashlib

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.retrieval.retriever import Citation, RetrievalFilter, _build_where, _document_to_citation, retrieve


class FakeEmbeddings(Embeddings):
    """Deterministic, offline embeddings for tests.

    Maps text to a fixed-length vector by hashing whitespace tokens into buckets
    (hashlib, so it's stable across processes). Semantic quality is irrelevant here:
    these tests assert metadata *filtering*, not ranking.
    """

    dim = 32

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in text.lower().split():
            bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        return vec

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def _meta(company, ticker, year, section, idx):
    return {
        "company": company,
        "ticker": ticker,
        "form_type": "10-K",
        "fiscal_year": year,
        "section": section,
        "source_url": f"https://sec.gov/{ticker}/{year}.htm",
        "char_start": idx * 100,
        "char_end": idx * 100 + 50,
        "chunk_index": idx,
    }


@pytest.fixture
def fake_store(tmp_path, monkeypatch):
    """A tiny Chroma over 4 known chunks (2 tickers x 2 years), fake embeddings."""
    from langchain_chroma import Chroma

    store = Chroma(
        collection_name="test_retriever",
        embedding_function=FakeEmbeddings(),
        persist_directory=str(tmp_path),
    )
    rows = [
        ("NVIDIA Corporation", "NVDA", 2023, "Risk Factors", "supply chain risk competition"),
        ("NVIDIA Corporation", "NVDA", 2024, "MD&A", "revenue grew driven by data center"),
        ("Apple Inc.", "AAPL", 2023, "Risk Factors", "supply chain risk competition"),
        ("Apple Inc.", "AAPL", 2024, "MD&A", "services revenue increased year over year"),
    ]
    texts = [text for *_, text in rows]
    metas = [_meta(co, tk, yr, sec, i) for i, (co, tk, yr, sec, _) in enumerate(rows)]
    ids = [f"{tk}-{yr}-10-k-{i}" for i, (_, tk, yr, _, _) in enumerate(rows)]
    store.add_texts(texts=texts, metadatas=metas, ids=ids)

    # retrieve() calls the module-level get_vectorstore; point it at our fake store.
    monkeypatch.setattr("src.retrieval.retriever.get_vectorstore", lambda settings=None: store)
    return store


def test_retrieve_unfiltered_returns_citations_across_filings(fake_store):
    cites = retrieve("revenue and risk", k=10)
    assert cites, "expected some citations"
    assert all(isinstance(c, Citation) for c in cites)
    tickers = {c.ticker for c in cites}
    assert tickers == {"NVDA", "AAPL"}  # unfiltered spans both companies


def test_retrieve_filters_by_ticker(fake_store):
    cites = retrieve("revenue and risk", filter=RetrievalFilter(tickers=["NVDA"]), k=10)
    assert cites
    assert all(c.ticker == "NVDA" for c in cites)


def test_retrieve_filters_by_ticker_and_year(fake_store):
    cites = retrieve(
        "revenue", filter=RetrievalFilter(tickers=["NVDA"], fiscal_years=[2024]), k=10
    )
    assert cites
    assert all(c.ticker == "NVDA" and c.fiscal_year == 2024 for c in cites)
    assert {c.section for c in cites} == {"MD&A"}  # NVDA 2024 chunk is the MD&A one


def test_no_filter_returns_none():
    assert _build_where(None) is None
    assert _build_where(RetrievalFilter()) is None


def test_single_field_uses_in_clause():
    where = _build_where(RetrievalFilter(tickers=["NVDA"]))
    assert where == {"ticker": {"$in": ["NVDA"]}}


def test_fiscal_years_stay_int_in_clause():
    where = _build_where(RetrievalFilter(fiscal_years=[2023, 2024]))
    assert where == {"fiscal_year": {"$in": [2023, 2024]}}
    assert all(isinstance(y, int) for y in where["fiscal_year"]["$in"])


def test_multiple_fields_combine_with_and():
    where = _build_where(RetrievalFilter(tickers=["NVDA"], fiscal_years=[2024]))
    assert where == {
        "$and": [
            {"ticker": {"$in": ["NVDA"]}},
            {"fiscal_year": {"$in": [2024]}},
        ]
    }


def test_empty_lists_are_ignored():
    # An explicitly-empty list means "no constraint", same as None.
    assert _build_where(RetrievalFilter(tickers=[], sections=[])) is None


def test_document_maps_to_citation_with_all_fields():
    doc = Document(
        page_content="TOKEN_RISK macroeconomic conditions and competition.",
        id="NVDA-2024-10-k-risk-factors-42",
        metadata={
            "company": "NVIDIA Corporation",
            "ticker": "NVDA",
            "form_type": "10-K",
            "fiscal_year": 2024,
            "section": "Risk Factors",
            "source_url": "https://www.sec.gov/Archives/edgar/data/1045810/x.htm",
            "char_start": 1000,
            "char_end": 1052,
            "chunk_index": 42,
        },
    )
    c = _document_to_citation(doc, score=0.23)
    assert c == Citation(
        company="NVIDIA Corporation",
        ticker="NVDA",
        form_type="10-K",
        fiscal_year=2024,
        section="Risk Factors",
        chunk_id="NVDA-2024-10-k-risk-factors-42",
        char_start=1000,
        char_end=1052,
        source_url="https://www.sec.gov/Archives/edgar/data/1045810/x.htm",
        snippet="TOKEN_RISK macroeconomic conditions and competition.",
        score=0.23,
    )
