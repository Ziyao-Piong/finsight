"""Section-aware chunking with rich metadata — the heart of Phase 2.

Phase 1 chunked the whole filing as one undifferentiated string, so a single chunk
could begin in Risk Factors and end in the MD&A, and nothing recorded *which* filing
or section a chunk came from. That makes filtered retrieval and citations impossible.

This module fixes both problems:

* **Chunk within a section, never across one.** We run the text splitter on each
  :class:`~src.ingest.parse.Section` independently, so a chunk's content is always
  homogeneous — it belongs to exactly one Item.
* **Attach queryable metadata to every chunk.** Each chunk records the filing it came
  from and where: ``{company, ticker, form_type, fiscal_year, section, accession,
  source_url, char_start, char_end, chunk_index}``. Phase 3 filters retrieval on these
  fields and renders citations from them; Phase 4 compares across filings using them.

Chroma only stores scalar metadata values, which is why the character span is two ints
(``char_start`` / ``char_end``) rather than a tuple.

The chunker is a pure function of its inputs (it takes ``chunk_size`` / ``chunk_overlap``
explicitly rather than reaching for global settings) so it is trivial to unit-test and
``store.py`` stays the one place that reads configuration.
"""

from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingest.parse import Section

# The metadata contract every chunk satisfies. Exposed so tests (and later phases)
# can assert against the schema in one place instead of hard-coding key lists.
REQUIRED_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "company",
        "ticker",
        "form_type",
        "fiscal_year",
        "section",
        "accession",
        "source_url",
        "char_start",
        "char_end",
        "chunk_index",
    }
)


def _slug(value: str) -> str:
    """Make a value safe for use inside a chunk id (lowercase, no separators)."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def chunk_filing(
    sections: list[Section],
    *,
    company: str,
    ticker: str,
    form_type: str,
    fiscal_year: int,
    accession: str,
    source_url: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """Split each section into chunks and tag every chunk with filing metadata.

    Args:
        sections: Labelled sections from :func:`src.ingest.parse.segment_sections`.
        company, ticker, form_type, fiscal_year, accession, source_url: Filing identity
            copied onto every chunk's metadata for filtering and citations.
        chunk_size, chunk_overlap: Character-based splitter knobs (``store.py`` passes
            these from :class:`~src.config.Settings`).

    Returns:
        A list of ``{"text", "metadata", "id"}`` dicts in document order. Ids are
        deterministic (``TICKER-YEAR-FORM-SECTION-INDEX``) so re-ingesting upserts in
        place instead of duplicating.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    form_slug = _slug(form_type)

    chunks: list[dict] = []
    for section in sections:
        pieces = splitter.split_text(section.text)
        section_slug = _slug(section.label)
        # Walk a cursor through the section so each piece's character span points back
        # into the original filing text (handy for "show me the source" later).
        cursor = 0
        for piece in pieces:
            local = section.text.find(piece, cursor)
            if local == -1:  # splitter stripped/normalised; fall back to the cursor
                local = cursor
            char_start = section.char_start + local
            char_end = char_start + len(piece)
            cursor = local + 1  # advance past this piece's start (chunks may overlap)

            chunk_index = len(chunks)
            chunks.append(
                {
                    "text": piece,
                    "id": f"{ticker}-{fiscal_year}-{form_slug}-{section_slug}-{chunk_index}",
                    "metadata": {
                        "company": company,
                        "ticker": ticker,
                        "form_type": form_type,
                        "fiscal_year": int(fiscal_year),
                        "section": section.label,
                        "accession": accession,
                        "source_url": source_url,
                        "char_start": char_start,
                        "char_end": char_end,
                        "chunk_index": chunk_index,
                    },
                }
            )
    return chunks
