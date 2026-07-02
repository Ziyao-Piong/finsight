# Phase 3 — Better Retrieval & Citations

## Goal
Make retrieval accurate and answers verifiable: filter the corpus by metadata before
searching, and carry each passage's source locators through to the answer so it can cite
exact, correctly-attributed passages.

## What we built
- `src/retrieval/retriever.py` — the retrieval layer and the single owner of vector-store
  access:
  - `RetrievalFilter` — an explicit set of metadata constraints (tickers, fiscal years,
    sections, form types).
  - `_build_where` — translates that filter into Chroma's `where` query language.
  - `Citation` — a retrieved passage plus its source locators.
  - `retrieve()` — metadata-filtered semantic search returning `Citation`s.
- `src/rag.py` — `ask()` now retrieves via `retrieve()` and returns an `Answer`
  (text + citations) instead of a bare string.
- `scripts/ask.py` — prints a numbered **Sources** block after the answer.

## Concepts

### Why metadata filtering matters
With one filing, semantic similarity alone is fine. With a multi-company, multi-year
corpus (Phase 2), a question about NVIDIA's 2024 revenue can pull visually-similar
passages from Apple's 2023 MD&A. Filtering on scalar metadata (`ticker`, `fiscal_year`,
`section`, `form_type`) narrows the search space *before* similarity ranking, so the
model only ever sees passages from the filings you meant.

### How a Chroma `where` clause works
Chroma filters on scalar metadata with a small query language: `{"ticker": {"$in":
["NVDA"]}}` keeps only chunks whose `ticker` is one of the listed values. To constrain on
more than one field you must wrap the conditions in `{"$and": [...]}` — a bare multi-key
dict is not valid. `_build_where` encodes exactly these rules, and returns `None` (no
filter) when nothing is constrained.

### The citation contract
Every chunk was tagged at ingest time (Phase 2) with `{company, ticker, form_type,
fiscal_year, section, source_url, char_start, char_end}` and a deterministic id. Retrieval
reads those straight off the returned document into a `Citation`, so the answer can show
*which filing, which section, which character span* each claim came from. `char_start`/
`char_end` point back into the cleaned filing text — the hook a future UI uses to render
an expandable, highlighted source.

### Why filters are explicit (and query understanding is deferred)
`retrieve()` does not parse "How did NVIDIA's revenue change FY23→FY24?" into filters —
the caller passes them. That parsing is *query understanding*, which is the Phase 4
agent's job (it chains `retrieve_filings` per company/year, then `extract_financials`,
then `calculate`). Keeping the retriever a pure "search with these filters" function keeps
it small, testable, and reusable by the agent without rework.

## Run it
    python -m scripts.ask "What were the main risk factors?"
Watch the answer stream, then read the numbered Sources block beneath it.

## Verify
    python -m pytest tests/test_retriever.py -q
The pure tests pin the filter→`where` translation and the document→`Citation` mapping; the
integration test builds a tiny Chroma with deterministic fake embeddings and proves a
ticker/year filter actually restricts the results — no network, no API keys.

## Design rationale / rough edges
- Retrieval quality levers (MMR, cross-encoder reranking, hybrid search) are deliberately
  *not* here — they're the "one deliberate improvement" measured with a before/after delta
  in Phase 5, the interview differentiator.
- `get_vectorstore` moved from `src/rag.py` into `src/retrieval/` this phase: the retrieval
  package is the natural owner of store access, and it removes the awkward seam where
  ingestion reached into the Phase-1 RAG glue for it.
