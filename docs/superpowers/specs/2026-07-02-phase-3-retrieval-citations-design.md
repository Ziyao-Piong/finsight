# Phase 3 — Better Retrieval & Citations (Design)

> **Status:** approved design, ready for implementation planning.
> Fits into `agents/plan.md` Phase 3 (4–6h): *"Make retrieval accurate and answers verifiable."*

## Goal

Replace Phase 1's naive `similarity_search` (see [`src/rag.py`](../../../src/rag.py)) with a
real retrieval layer that supports **metadata filtering** (by ticker / fiscal year / section /
form type) and **carries source locators through to the answer** so the response can render
expandable, correctly-attributed citations.

**Milestone:** answers cite exact, correctly-attributed passages.
**Verify:** `pytest tests/test_retriever.py` — relevant passages returned with correct metadata (TDD).

## Settled design decisions

1. **Filters are explicit, not inferred.** `retrieve()` accepts a structured filter supplied by
   the caller. Turning a natural-language question into filters (query understanding) is
   **deferred to the Phase 4 agent**, which the plan already scopes as chaining `retrieve_filings`
   per company/year. This keeps Phase 3 tightly bounded and cleanly testable, and avoids building
   the agent's job twice.
2. **Results are structured `Citation` objects.** The retriever returns typed citations, not raw
   LangChain `Document`s and not inline `[1]`/`[2]` markers parsed out of prose. This is the stable
   contract the Phase 6 API/web layer renders as expandable sources, and it is directly assertable
   in tests.
3. **No retrieval-quality technique yet.** MMR / cross-encoder reranking / hybrid search are
   **deferred to Phase 5**, where they become the "one deliberate improvement" measured with a
   before/after delta (the interview differentiator). Phase 3 = metadata filtering + citations only.

## Architecture

### New package: `src/retrieval/`

`src/retrieval/__init__.py` + `src/retrieval/retriever.py`. Three single-purpose units:

#### `RetrievalFilter` (dataclass) — the explicit filter interface

```python
@dataclass(frozen=True)
class RetrievalFilter:
    tickers: list[str] | None = None
    fiscal_years: list[int] | None = None
    sections: list[str] | None = None
    form_types: list[str] | None = None
```

All fields optional; `None` or empty means "no constraint on that field." Field names map onto the
chunk metadata written by [`src/ingest/chunk.py`](../../../src/ingest/chunk.py)
(`ticker`, `fiscal_year`, `section`, `form_type`).

#### `Citation` (dataclass) — the structured result, one per retrieved chunk

```python
@dataclass(frozen=True)
class Citation:
    company: str
    ticker: str
    form_type: str
    fiscal_year: int
    section: str
    chunk_id: str
    char_start: int
    char_end: int
    source_url: str
    snippet: str      # the chunk text, used to render the expandable source
    score: float | None = None  # similarity distance from Chroma, if available
```

Built directly from a chunk's metadata (which `chunk.py` already carries in full) plus the chunk
text as `snippet`. Every field the plan's citation spec lists
(`{company, form_type, fiscal_year, section, chunk_id, char_span}`) is present, plus `source_url`
and `snippet` for rendering.

#### `_build_where(filter) -> dict | None` — pure Chroma-query translation (the heart of the TDD)

Translates a `RetrievalFilter` into Chroma's `where` syntax:

- No filter / all fields `None` → returns `None` (unfiltered search).
- One active field → `{"<field>": {"$in": [values...]}}`.
- Multiple active fields → `{"$and": [<clause>, <clause>, ...]}` (Chroma requires `$and` to
  combine conditions; a bare multi-key dict is not valid).
- `fiscal_years` values stay `int` (metadata stores them as int — see `chunk.py`).

Pure function of its input, no network, no settings — trivially unit-testable.

#### `retrieve(query, *, filter=None, k=None, settings=None) -> list[Citation]`

Reuses `get_vectorstore(settings)` (the same handle ingestion writes through, so collection/embedding/persist-dir can never disagree). Calls `similarity_search_with_score(query, k=k, filter=_build_where(filter))`, maps each `(Document, score)` to a `Citation`. `k` defaults to `settings.retrieval_k`.

### Changes to `src/rag.py`

- `ask()` gains an optional `filter: RetrievalFilter | None` param and now returns a small
  **`Answer`** dataclass (`text: str`, `citations: list[Citation]`) instead of a bare `str`.
  This is a deliberate breaking change; the only consumer is `scripts/ask.py`.
- `ask()` calls `retrieve()` instead of `store.similarity_search`, formats the context block with
  each passage numbered `[1]…[k]` aligned to the citation list (so grounding is preserved and a
  future UI can line up prose to sources), generates as before (streaming unchanged), and attaches
  the citations to the returned `Answer`.
- `_format_context` is updated to number passages consistently with the citation ordering.
- The "no chunks found" path returns an `Answer` with an empty citation list and the guidance message.

### Changes to `scripts/ask.py`

After streaming the answer, print a **Sources** section listing each citation, e.g.:

```
Sources
[1] NVIDIA Corp. 10-K FY2024 — Risk Factors  (NVDA-2024-10-k-risk-factors-42)
[2] NVIDIA Corp. 10-K FY2024 — MD&A          (NVDA-2024-10-k-mda-91)
```

This makes the phase milestone ("answers cite exact, correctly-attributed passages") visible from
the CLI. UTF-8 stdout reconfiguration already in place stays.

## Testing (TDD — `tests/test_retriever.py`)

Follows `tests/conftest.py`'s offline/synthetic philosophy (no network, no real SEC HTML, no API keys):

1. **`_build_where` unit tests** (pure): no filter → `None`; single field → `$in` clause;
   multiple fields → `$and` of clauses; `fiscal_years` remain `int`.
2. **`Citation` mapping test**: a `Document` with known metadata + text → a `Citation` with every
   field correctly populated.
3. **Integration test** proving metadata filtering *actually restricts results*: a **fake
   deterministic embeddings** class (implements the LangChain `Embeddings` interface, maps text to a
   stable vector with no network) builds a tiny tmp-dir Chroma from a handful of known chunks with
   distinct metadata (e.g. two tickers × two years). Assert that filtering by ticker and by year
   returns only citations whose metadata matches, and that unfiltered search can return across all.
   The fake embeddings fixture lives in the test module (or `conftest.py` if reused later).

The plan's Phase 3 verify step — "relevant passages with correct metadata" — is an integration
behavior, so the integration test is included rather than testing only the pure pieces.

## Documentation

`phase/phase_3.md` — a teaching-oriented phase guide matching the `phase_0`–`phase_2` style
(explains *why*, not just *what*): why metadata filtering matters for multi-filing corpora, how
Chroma `where` clauses / `$in` / `$and` work, the citation contract and how it feeds the future
API/web layer, and why query→filter understanding is deferred to the Phase 4 agent.

Update `agents/plan.md` status line and `CLAUDE.md` "Current status" to reflect Phase 3 complete
once built.

## Out of scope (YAGNI / deferred)

- **MMR / reranking / hybrid search** → Phase 5 (measured improvement).
- **Query → filter parsing / query rewriting** → Phase 4 agent.
- **New dependencies** — none; Chroma already supports `where` filtering and `similarity_search_with_score`.
