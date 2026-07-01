"""Tests for src/ingest/chunk.py — section-aware chunking + metadata (TDD, first).

These pin down the Phase 2 core promise: chunks never cross a section boundary, and
every chunk carries complete, correct metadata so later phases can filter and cite.
We build Section objects directly (rather than parsing HTML) so a parse bug can't make
these fail — the unit under test is the chunker.
"""

from __future__ import annotations

from src.ingest.chunk import REQUIRED_METADATA_KEYS, chunk_filing
from src.ingest.parse import Section

# Common per-filing identity passed to chunk_filing in every test.
_FILING_KWARGS = dict(
    company="Apple Inc.",
    ticker="AAPL",
    form_type="10-K",
    fiscal_year=2023,
    accession="0000320193-23-000106",
    source_url="https://www.sec.gov/Archives/edgar/data/320193/.../aapl-20230930.htm",
    chunk_size=120,
    chunk_overlap=20,
)


def _make_sections() -> list[Section]:
    # Each section's text is long enough that a 120-char chunk_size yields several
    # chunks, proving we split *within* a section. Unique marker words let us assert
    # which body text landed where.
    risk_text = "Item 1A. Risk Factors\n" + ("RISKWORD " * 60)
    mda_text = "Item 7. MD&A\n" + ("MDAWORD " * 60)
    risk = Section("1A", "Risk Factors", risk_text, 1000, 1000 + len(risk_text))
    mda = Section("7", "MD&A", mda_text, 5000, 5000 + len(mda_text))
    return [risk, mda]


def test_every_chunk_has_complete_correct_metadata():
    chunks = chunk_filing(_make_sections(), **_FILING_KWARGS)
    assert chunks, "expected at least one chunk"
    for c in chunks:
        meta = c["metadata"]
        assert REQUIRED_METADATA_KEYS <= set(meta), f"missing keys: {REQUIRED_METADATA_KEYS - set(meta)}"
        assert meta["company"] == "Apple Inc."
        assert meta["ticker"] == "AAPL"
        assert meta["form_type"] == "10-K"
        assert meta["accession"] == "0000320193-23-000106"


def test_fiscal_year_is_stored_as_int():
    chunks = chunk_filing(_make_sections(), **_FILING_KWARGS)
    assert all(isinstance(c["metadata"]["fiscal_year"], int) for c in chunks)
    assert all(c["metadata"]["fiscal_year"] == 2023 for c in chunks)


def test_chunks_never_cross_a_section_boundary():
    chunks = chunk_filing(_make_sections(), **_FILING_KWARGS)
    for c in chunks:
        if c["metadata"]["section"] == "Risk Factors":
            assert "MDAWORD" not in c["text"]
        if c["metadata"]["section"] == "MD&A":
            assert "RISKWORD" not in c["text"]
    # Both sections actually produced multiple chunks (within-section splitting).
    sections_seen = {c["metadata"]["section"] for c in chunks}
    assert sections_seen == {"Risk Factors", "MD&A"}
    assert sum(c["metadata"]["section"] == "Risk Factors" for c in chunks) > 1


def test_ids_are_unique_deterministic_and_descriptive():
    chunks = chunk_filing(_make_sections(), **_FILING_KWARGS)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    # Deterministic: re-chunking the same input yields the same ids (idempotent ingest).
    again = [c["id"] for c in chunk_filing(_make_sections(), **_FILING_KWARGS)]
    assert ids == again
    # Descriptive: id encodes the filing identity so it reads in the vector store.
    assert all(i.startswith("AAPL-2023-") for i in ids)


def test_char_spans_sit_within_their_section():
    sections = _make_sections()
    by_label = {s.label: s for s in sections}
    chunks = chunk_filing(sections, **_FILING_KWARGS)
    for c in chunks:
        meta = c["metadata"]
        sec = by_label[meta["section"]]
        assert sec.char_start <= meta["char_start"] < meta["char_end"] <= sec.char_end
