"""Tests for src/ingest/parse.py — section segmentation (written first, TDD).

The single behaviour these tests pin down: given cleaned 10-K text whose Item
headings appear both in the table of contents and in the body, segment_sections
must return the *body* sections (not the TOC), correctly labelled and in order,
with non-overlapping character spans.
"""

from __future__ import annotations

from src.ingest.parse import SECTION_LABELS, html_to_text, segment_sections


def test_segments_the_four_body_items_in_order(synthetic_filing_text):
    sections = segment_sections(synthetic_filing_text)
    assert [s.item for s in sections] == ["1", "1A", "7", "8"]


def test_known_items_get_friendly_labels(synthetic_filing_text):
    by_item = {s.item: s for s in segment_sections(synthetic_filing_text)}
    assert by_item["1A"].label == "Risk Factors"
    assert by_item["7"].label == "MD&A"
    assert by_item["8"].label == "Financial Statements"
    assert SECTION_LABELS["1A"] == "Risk Factors"


def test_section_holds_its_own_body_and_not_the_next(synthetic_filing_text):
    by_item = {s.item: s for s in segment_sections(synthetic_filing_text)}
    risk = by_item["1A"]
    # The Risk Factors section owns its body marker...
    assert "TOKEN_RISK" in risk.text
    # ...and must NOT bleed into the neighbouring sections' bodies.
    assert "TOKEN_MDA" not in risk.text
    assert "TOKEN_BUSINESS" not in risk.text


def test_toc_is_not_emitted_as_sections(synthetic_filing_text):
    sections = segment_sections(synthetic_filing_text)
    # Exactly four body sections — the TOC duplicates are discarded, not doubled.
    assert len(sections) == 4
    # The header line ("Item 1A. ...") and body are kept; page-number TOC noise
    # ("Risk Factors 10") sits before the first body section and is dropped.
    risk = next(s for s in sections if s.item == "1A")
    assert "Risk Factors 10" not in risk.text


def test_html_to_text_normalizes_nonbreaking_spaces():
    # Real SEC filings (e.g. NVIDIA's) render Item headings with a non-breaking space:
    # "Item&nbsp;8.". If we don't normalise it, the line-anchored heading regex misses
    # the body heading entirely and the section is lost. Regression test for that bug.
    text = html_to_text("<p>Item&#160;8. Financial Statements</p>")
    assert "\xa0" not in text
    assert "Item 8. Financial Statements" in text


def test_segments_body_heading_written_with_nonbreaking_space():
    # TOC uses a normal space; the body heading uses NBSP (as NVIDIA's filings do).
    # After normalisation, the body occurrence must still win so Item 8 isn't truncated
    # to its one-line TOC entry.
    raw_html = (
        "<p>Item 8. Financial Statements 44</p>"  # TOC line, page number
        "<p>Item 1. Business</p><p>NVIDIA pioneered accelerated computing.</p>"
        "<p>Item&#160;8. Financial Statements and Supplementary Data</p>"
        "<p>TOKEN_FIN total revenue was reported here.</p>"
    )
    sections = segment_sections(html_to_text(raw_html))
    fin = next(s for s in sections if s.item == "8")
    assert "TOKEN_FIN" in fin.text  # body content, not the truncated TOC entry


def test_char_spans_are_ordered_and_match_the_source(synthetic_filing_text):
    text = synthetic_filing_text
    sections = segment_sections(text)
    prev_end = -1
    for s in sections:
        assert 0 <= s.char_start < s.char_end <= len(text)
        # Span actually slices the section's own text out of the source document.
        assert text[s.char_start : s.char_end] == s.text
        # Non-overlapping and strictly increasing.
        assert s.char_start >= prev_end
        prev_end = s.char_end
