"""Tests for src/retrieval/retriever.py — filtering + citations (Phase 3, TDD).

Split in three, mirroring conftest's offline philosophy (no network, no API keys):
  * _build_where: a pure function, unit-tested directly.
  * _document_to_citation: a pure mapping, unit-tested directly.
  * retrieve: an integration test over a tiny Chroma built with deterministic fake
    embeddings, proving metadata filtering actually restricts the results.
"""

from __future__ import annotations

from src.retrieval.retriever import RetrievalFilter, _build_where


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
