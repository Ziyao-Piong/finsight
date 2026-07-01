"""Phase 2 ingestion pipeline: EDGAR fetch -> parse -> section-aware chunk -> store.

This package replaces Phase 1's single-file, metadata-free ingest (one hardcoded
Apple 10-K) with a real multi-document pipeline. The four modules each own one step:

  edgar.py   discover + fetch filings from SEC EDGAR by ticker/year
  parse.py   HTML -> clean text, then segment into labelled sections (Item 1A, 7, 8 ...)
  chunk.py   split each section into chunks that carry rich, queryable metadata
  store.py   orchestrate the above and persist chunks (+ metadata + ids) to Chroma

The metadata each chunk carries here -- {company, ticker, form_type, fiscal_year,
section, char span} -- is what later phases stand on: Phase 3 filters retrieval by it
and renders citations from it, Phase 4 compares across documents using it, and Phase 5
evaluates against it.
"""
