# Phase 3 — Better Retrieval & Citations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Phase 1's naive `similarity_search` with a metadata-filtered retrieval layer that returns structured `Citation` objects, so answers cite exact, correctly-attributed passages.

**Architecture:** A new `src/retrieval` package owns all vector-store access. `retriever.py` exposes an explicit `RetrievalFilter`, a `retrieve()` function that runs metadata-filtered semantic search, and a `Citation` result type. `src/rag.py`'s `ask()` is rewired to use it and now returns an `Answer` (text + citations). Query→filter understanding is deferred to the Phase 4 agent; reranking to Phase 5.

**Tech Stack:** Python, LangChain (`langchain-chroma`, `langchain-core`), Chroma vector store, pytest. No new dependencies.

## Global Constraints

- **Run modules with `python -m <module>`**, never `python path/to/file.py` (repo-root `sys.path`).
- **`src/config.py` is the only reader of `os.environ` for app config**; everything else calls `get_settings()`.
- **`src/llm/factory.py` is the only module that names a concrete vendor.** Retrieval stays provider-agnostic (touches only `get_embeddings` / `get_chat_model` indirectly).
- **Chroma stores only scalar metadata.** Filters operate on scalar fields (`ticker`, `fiscal_year:int`, `section`, `form_type`).
- **Teaching-oriented docstrings**: explain *why* a design choice was made, matching the existing `src/` style.
- **Commits hide the assistant**: do NOT add any `Co-Authored-By` / assistant attribution trailer (repo rule "hide yourself when commit").
- **TDD**: write the failing test first, watch it fail, implement minimally, watch it pass, commit.

---

### Task 1: Scaffold the retrieval package and relocate `get_vectorstore`

`get_vectorstore` currently lives in `src/rag.py`, and `src/ingest/store.py`'s docstring already
notes it "reaches into" rag for it. Its natural home is the retrieval package (the single owner of
vector-store access). Moving it here also avoids a circular import once `rag.py` starts importing
from `retrieval`.

**Files:**
- Create: `src/retrieval/__init__.py`
- Create: `src/retrieval/retriever.py`
- Modify: `src/rag.py` (remove `get_vectorstore` def; import it from retrieval)
- Modify: `src/ingest/store.py:28` (import `get_vectorstore` from retrieval; fix docstring reference)

**Interfaces:**
- Produces: `src.retrieval.retriever.get_vectorstore(settings: Settings | None = None) -> Chroma`
  — same behavior and signature as the current `src.rag.get_vectorstore`.

- [ ] **Step 1: Create the package init**

Create `src/retrieval/__init__.py`:

```python
"""Retrieval layer: metadata-filtered semantic search over the filing corpus.

This package is the single owner of vector-store access (``get_vectorstore``) and
turns a question + explicit filters into structured, citable results. Phase 4's
agent will decide *which* filters to pass; Phase 3 just retrieves and cites.
"""
```

- [ ] **Step 2: Create `src/retrieval/retriever.py` with the relocated `get_vectorstore`**

Create `src/retrieval/retriever.py` (the rest of the module is added in later tasks):

```python
"""Metadata-filtered semantic search and structured citations — the heart of Phase 3.

Phase 1 retrieved with a bare ``similarity_search``: no metadata filtering, and the
answer had no idea which filing/section a passage came from, so it couldn't cite. This
module fixes both. It owns vector-store access for the whole app (:func:`get_vectorstore`),
translates an explicit :class:`RetrievalFilter` into a Chroma ``where`` clause, and returns
:class:`Citation` objects carrying the source locators the answer renders as expandable
sources.

Why *explicit* filters (not parsed from the question)? Turning "How did NVIDIA's revenue
change FY23→FY24?" into ``tickers=['NVDA'], fiscal_years=[2023, 2024]`` is query
understanding — the Phase 4 agent's job, which chains ``retrieve_filings`` per company/year.
Keeping Phase 3's retriever a pure "search with these filters" function keeps it small and
testable and avoids building the agent twice.
"""

from __future__ import annotations

from src.config import Settings, get_settings


def get_vectorstore(settings: Settings | None = None):
    """Return the persistent Chroma vector store for the current provider.

    Both ingestion (writing chunks) and querying (reading them) go through this one
    function so they always agree on three things: the persist directory, the embedding
    function, and the collection name. The collection name is keyed by provider (see
    ``Settings.collection_name``) because each provider's embeddings live in their own,
    incomparable vector space.
    """
    settings = settings or get_settings()
    # Imported lazily so importing this module doesn't require chromadb until you
    # actually build or query an index (mirrors the factory's lazy-import style).
    from langchain_chroma import Chroma

    from src.llm.factory import get_embeddings

    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings),
        persist_directory=settings.chroma_dir,
    )
```

- [ ] **Step 3: Remove `get_vectorstore` from `src/rag.py` and import it instead**

In `src/rag.py`, delete the `get_vectorstore` function (currently lines ~41–59) and its now-unused
`get_embeddings` import. Add the import near the other `src` imports:

```python
from src.config import Settings, get_settings
from src.llm.factory import get_chat_model
from src.retrieval.retriever import get_vectorstore
```

(Leave the rest of `rag.py` unchanged in this task — `ask()` still calls `get_vectorstore(settings)`.)

- [ ] **Step 4: Update `src/ingest/store.py` import and docstring**

Change the import at `src/ingest/store.py:28` from:

```python
from src.rag import get_vectorstore
```

to:

```python
from src.retrieval.retriever import get_vectorstore
```

In the module docstring, change the bullet that reads "We persist through
:func:`src.rag.get_vectorstore`, the *same* handle the query path uses" to
":func:`src.retrieval.retriever.get_vectorstore`".

- [ ] **Step 5: Verify imports resolve (no circular import) and the existing suite still passes**

Run:
```bash
python -c "import src.rag, src.ingest.store, src.retrieval.retriever; print('imports OK')"
python -m pytest tests/ -q
```
Expected: prints `imports OK`, then existing `test_parse.py` / `test_chunk.py` pass (no failures).

- [ ] **Step 6: Commit**

```bash
git add src/retrieval/__init__.py src/retrieval/retriever.py src/rag.py src/ingest/store.py
git commit -m "Phase 3: relocate get_vectorstore into src/retrieval package"
```

---

### Task 2: `RetrievalFilter` and the `_build_where` Chroma translation

The pure core of the phase: turn an explicit filter into Chroma's `where` syntax.

**Files:**
- Modify: `src/retrieval/retriever.py`
- Create: `tests/test_retriever.py`

**Interfaces:**
- Produces:
  - `RetrievalFilter(tickers: list[str] | None = None, fiscal_years: list[int] | None = None, sections: list[str] | None = None, form_types: list[str] | None = None)` — frozen dataclass.
  - `_build_where(filter: RetrievalFilter | None) -> dict | None` — pure function.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_retriever.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_retriever.py -q`
Expected: FAIL — `ImportError: cannot import name 'RetrievalFilter'` (and `_build_where`).

- [ ] **Step 3: Implement `RetrievalFilter` and `_build_where`**

Add to `src/retrieval/retriever.py` (after the imports, before `get_vectorstore`):

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalFilter:
    """Explicit metadata constraints for a retrieval call.

    Every field is optional; ``None`` or an empty list means "don't constrain on this
    field". The field names match the chunk metadata written by
    :mod:`src.ingest.chunk` (``ticker``, ``fiscal_year``, ``section``, ``form_type``).
    The caller supplies these — parsing them from a natural-language question is the
    Phase 4 agent's job.
    """

    tickers: list[str] | None = None
    fiscal_years: list[int] | None = None
    sections: list[str] | None = None
    form_types: list[str] | None = None


# Maps a RetrievalFilter field to the chunk-metadata key it constrains.
_FILTER_FIELDS: tuple[tuple[str, str], ...] = (
    ("tickers", "ticker"),
    ("fiscal_years", "fiscal_year"),
    ("sections", "section"),
    ("form_types", "form_type"),
)


def _build_where(filter: RetrievalFilter | None) -> dict | None:  # noqa: A002
    """Translate a :class:`RetrievalFilter` into a Chroma ``where`` clause.

    Chroma's query language uses ``$in`` for "value is one of" and requires ``$and`` to
    combine conditions on more than one field (a bare multi-key dict is not valid). So:
      * no active fields          -> ``None`` (unfiltered search)
      * exactly one active field  -> ``{field: {"$in": [...]}}``
      * two or more active fields -> ``{"$and": [clause, clause, ...]}``
    """
    if filter is None:
        return None

    clauses: list[dict] = []
    for attr, meta_key in _FILTER_FIELDS:
        values = getattr(filter, attr)
        if values:  # skip None and empty lists
            clauses.append({meta_key: {"$in": list(values)}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_retriever.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/retriever.py tests/test_retriever.py
git commit -m "Phase 3: RetrievalFilter + _build_where Chroma translation"
```

---

### Task 3: `Citation` and the `_document_to_citation` mapping

**Files:**
- Modify: `src/retrieval/retriever.py`
- Modify: `tests/test_retriever.py`

**Interfaces:**
- Produces:
  - `Citation(company, ticker, form_type, fiscal_year, section, chunk_id, char_start, char_end, source_url, snippet, score=None)` — frozen dataclass.
  - `_document_to_citation(doc: Document, score: float | None) -> Citation` — pure mapping.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_retriever.py` (imports and a new test):

```python
from langchain_core.documents import Document

from src.retrieval.retriever import Citation, _document_to_citation


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retriever.py::test_document_maps_to_citation_with_all_fields -q`
Expected: FAIL — `ImportError: cannot import name 'Citation'`.

- [ ] **Step 3: Implement `Citation` and `_document_to_citation`**

Add to `src/retrieval/retriever.py`. Put the `Citation` dataclass just below `RetrievalFilter`,
and the mapping helper below `_build_where`:

```python
@dataclass(frozen=True)
class Citation:
    """One retrieved passage plus the source locators needed to cite it.

    This is the stable contract the Phase 6 API/web layer renders as an expandable
    source. Built from a chunk's metadata (written by :mod:`src.ingest.chunk`) plus the
    chunk text as ``snippet``. ``char_start``/``char_end`` locate the passage in the
    cleaned filing text; ``score`` is Chroma's similarity distance (lower = closer),
    ``None`` when the caller didn't ask for scores.
    """

    company: str
    ticker: str
    form_type: str
    fiscal_year: int
    section: str
    chunk_id: str
    char_start: int
    char_end: int
    source_url: str
    snippet: str
    score: float | None = None
```

```python
def _document_to_citation(doc, score: float | None) -> Citation:
    """Map a retrieved LangChain ``Document`` (+ score) to a :class:`Citation`.

    ``doc.id`` is the deterministic chunk id Chroma stored at ingest time
    (``TICKER-YEAR-FORM-SECTION-INDEX``); the rest come from the chunk metadata.
    """
    meta = doc.metadata
    return Citation(
        company=meta["company"],
        ticker=meta["ticker"],
        form_type=meta["form_type"],
        fiscal_year=meta["fiscal_year"],
        section=meta["section"],
        chunk_id=doc.id,
        char_start=meta["char_start"],
        char_end=meta["char_end"],
        source_url=meta["source_url"],
        snippet=doc.page_content,
        score=score,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_retriever.py::test_document_maps_to_citation_with_all_fields -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/retriever.py tests/test_retriever.py
git commit -m "Phase 3: Citation type + document->citation mapping"
```

---

### Task 4: `retrieve()` — metadata-filtered search proven with fake embeddings

**Files:**
- Modify: `src/retrieval/retriever.py`
- Modify: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `get_vectorstore` (Task 1), `_build_where` (Task 2), `_document_to_citation` (Task 3).
- Produces: `retrieve(query: str, *, filter: RetrievalFilter | None = None, k: int | None = None, settings: Settings | None = None) -> list[Citation]`.

- [ ] **Step 1: Write the failing integration test**

Add to `tests/test_retriever.py` a deterministic fake-embeddings class, a fixture that builds a
tiny Chroma from four known chunks, and the filtering assertions:

```python
import hashlib

import pytest
from langchain_core.embeddings import Embeddings

from src.retrieval.retriever import retrieve


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_retriever.py -q`
Expected: the three `test_retrieve_*` FAIL with `ImportError: cannot import name 'retrieve'` (the pure-function tests from Tasks 2–3 still pass).

- [ ] **Step 3: Implement `retrieve()`**

Add to `src/retrieval/retriever.py` (below `get_vectorstore`):

```python
def retrieve(
    query: str,
    *,
    filter: RetrievalFilter | None = None,  # noqa: A002
    k: int | None = None,
    settings: Settings | None = None,
) -> list[Citation]:
    """Return the top-``k`` passages matching ``query`` under the given metadata filter.

    Embeds the query, runs a metadata-filtered similarity search over the provider's
    Chroma collection, and returns :class:`Citation` objects carrying the source locators
    an answer needs to cite. ``k`` defaults to ``settings.retrieval_k``.
    """
    settings = settings or get_settings()
    k = k or settings.retrieval_k

    store = get_vectorstore(settings)
    where = _build_where(filter)
    results = store.similarity_search_with_score(query, k=k, filter=where)
    return [_document_to_citation(doc, score) for doc, score in results]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_retriever.py -q`
Expected: PASS (all tests: pure + integration).

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/retriever.py tests/test_retriever.py
git commit -m "Phase 3: retrieve() metadata-filtered search returning Citations"
```

---

### Task 5: `Answer` type and rewire `ask()` to use `retrieve()`

**Files:**
- Modify: `src/rag.py`
- Create: `tests/test_rag.py`

**Interfaces:**
- Consumes: `retrieve`, `RetrievalFilter`, `Citation` (Task 4).
- Produces:
  - `Answer(text: str, citations: list[Citation])` — frozen dataclass.
  - `ask(question, settings=None, k=None, filter=None, stream=True) -> Answer` (return type changed from `str`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rag.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rag.py -q`
Expected: FAIL — `ImportError: cannot import name 'Answer'` (and `ask` returns a `str` today).

- [ ] **Step 3: Rewire `src/rag.py`**

Replace the imports block, drop the old `get_vectorstore` usage details, and rewrite
`_format_context` + `ask()`. The new module body (from the imports down) reads:

```python
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import Settings, get_settings
from src.llm.factory import get_chat_model
from src.retrieval.retriever import Citation, RetrievalFilter, get_vectorstore, retrieve

_SYSTEM_PROMPT = (
    "You are FinSight, a financial-filings assistant. Answer the user's question "
    "using ONLY the context passages provided below, which are excerpts from an SEC "
    "filing. If the answer is not contained in the context, say you don't have enough "
    "information from the filing to answer — do not use outside knowledge or guess. "
    "Be concise and factual."
)


@dataclass(frozen=True)
class Answer:
    """A grounded answer plus the passages it was drawn from.

    ``text`` is the model's answer; ``citations`` are the retrieved passages (in the same
    order they were numbered in the context) so a UI can render expandable sources.
    """

    text: str
    citations: list[Citation]


def _format_context(citations: list[Citation]) -> str:
    """Number each passage ``[i]`` so the model can attribute claims to a source.

    The numbering matches the order of ``citations`` on the returned :class:`Answer`, so a
    later UI can line prose up with the expandable source it came from.
    """
    return "\n\n".join(
        f"[{i}] {c.company} {c.form_type} FY{c.fiscal_year} — {c.section}\n{c.snippet}"
        for i, c in enumerate(citations, start=1)
    )


def ask(
    question: str,
    settings: Settings | None = None,
    k: int | None = None,
    filter: RetrievalFilter | None = None,  # noqa: A002
    stream: bool = True,
) -> Answer:
    """Answer ``question`` from the ingested filings and return an :class:`Answer`.

    Args:
        question: The user's natural-language question.
        settings: Optional settings override (defaults to the shared cached settings).
        k: How many chunks to retrieve. Defaults to ``settings.retrieval_k``.
        filter: Optional metadata constraints (ticker / fiscal year / section / form).
            The Phase 4 agent will supply these; the CLI leaves it ``None``.
        stream: When True (default), print tokens to stdout as they arrive.

    Returns:
        An :class:`Answer` with the answer text and the citations it was grounded in.
    """
    settings = settings or get_settings()
    k = k or settings.retrieval_k

    # --- Retrieve -------------------------------------------------------------
    citations = retrieve(question, filter=filter, k=k, settings=settings)

    if not citations:
        msg = (
            "No chunks found in the vector store. Have you run "
            "`python -m scripts.ingest` for this provider yet?"
        )
        if stream:
            print(msg)
        return Answer(text=msg, citations=[])

    # --- Augment --------------------------------------------------------------
    context = _format_context(citations)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]

    # --- Generate -------------------------------------------------------------
    chat = get_chat_model(settings)
    parts: list[str] = []
    for chunk in chat.stream(messages):
        text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
        parts.append(text)
        if stream:
            print(text, end="", flush=True)
    if stream:
        print()  # final newline after the streamed answer

    return Answer(text="".join(parts), citations=citations)
```

Note: keep the module docstring at the top of `rag.py` as-is (it still describes the RAG loop);
only the code below the docstring changes. `get_vectorstore` is re-exported via the import so any
external `from src.rag import get_vectorstore` still resolves.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_rag.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the whole suite (nothing regressed)**

Run: `python -m pytest tests/ -q`
Expected: PASS (parse, chunk, retriever, rag).

- [ ] **Step 6: Commit**

```bash
git add src/rag.py tests/test_rag.py
git commit -m "Phase 3: ask() uses retrieve(), returns Answer with citations"
```

---

### Task 6: Render citations in the CLI (`scripts/ask.py`)

**Files:**
- Modify: `scripts/ask.py`
- Modify: `tests/test_rag.py` (test the pure formatter)

**Interfaces:**
- Consumes: `Answer`, `Citation`, `ask` (Task 5).
- Produces: `format_sources(citations: list[Citation]) -> str` in `scripts/ask.py`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_rag.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rag.py -k format_sources -q`
Expected: FAIL — `ImportError: cannot import name 'format_sources'`.

- [ ] **Step 3: Update `scripts/ask.py`**

Rewrite `scripts/ask.py` to add the formatter and print sources after the streamed answer:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_rag.py -k format_sources -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/ask.py tests/test_rag.py
git commit -m "Phase 3: render numbered source citations in the ask CLI"
```

---

### Task 7: Documentation — phase guide and status updates

**Files:**
- Create: `phase/phase_3.md`
- Modify: `agents/plan.md:3-5` (status line) and `agents/plan.md:140-146` (mark Phase 3 done)
- Modify: `CLAUDE.md` ("Current status" paragraph under "What this is")

**Interfaces:** none (docs only).

- [ ] **Step 1: Write the phase guide**

Create `phase/phase_3.md` matching the teaching style of `phase_0`–`phase_2` (explain *why*).
Include these sections with real content:

```markdown
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
```

- [ ] **Step 2: Update `agents/plan.md` status**

Change the status blockquote near the top (lines ~3–5) from "Phases 0–2 built ... Phases 3–7 not
yet started." to reflect Phase 3 built:

```markdown
> **Status:** Phases 0–3 built (Foundations, walking-skeleton RAG, production-shaped
> ingestion, metadata-filtered retrieval + citations). Phases 4–7 not yet started.
> This is the living project plan. Phase-specific learning guides live in `phase/`.
```

Under "### Phase 3 — Better retrieval & citations", append a line marking it done:

```markdown
- **Status: ✅ Built.** `src/retrieval/retriever.py` (`RetrievalFilter`, `_build_where`,
  `Citation`, `retrieve`); `ask()` returns an `Answer` with citations; `tests/test_retriever.py`.
```

- [ ] **Step 3: Update `CLAUDE.md` current status**

In the "What this is" section, change the "**Current status: Phase 2 complete**" sentence to:

```markdown
**Current status: Phase 3 complete** (metadata-filtered retrieval + citations). Built so
far: Phase 0 skeleton, Phase 1 walking-skeleton RAG, Phase 2's `src/ingest/` pipeline, and
Phase 3's `src/retrieval/retriever.py` (`RetrievalFilter`, `_build_where`, `Citation`,
`retrieve`) with `ask()` returning an `Answer` (text + citations) and
`tests/test_retriever.py` / `tests/test_rag.py`. Still **not built yet**: `agent/`, `api/`,
`web/`, `eval/`.
```

- [ ] **Step 4: Verify docs render and the full suite passes**

Run: `python -m pytest tests/ -q`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add phase/phase_3.md agents/plan.md CLAUDE.md
git commit -m "Phase 3: phase guide + status updates"
```

---

## Self-Review

**Spec coverage:**
- Metadata-filtered retrieval → Tasks 2 (`_build_where`) + 4 (`retrieve`). ✅
- Structured `Citation` objects → Task 3. ✅
- Explicit filters, query understanding deferred → `RetrievalFilter` (Task 2), documented in phase guide (Task 7). ✅
- `ask()` returns `Answer` (text + citations) → Task 5. ✅
- CLI renders expandable/attributed sources → Task 6. ✅
- TDD via `tests/test_retriever.py` (+ `test_rag.py`) with offline fake embeddings → Tasks 2–6. ✅
- No reranking/MMR, no new deps → honored across all tasks. ✅
- `phase/phase_3.md` + status updates → Task 7. ✅
- `get_vectorstore` relocation (design refinement consistent with "reuses get_vectorstore") → Task 1. ✅

**Placeholder scan:** No TBD/TODO/"add error handling"/"similar to Task N". Every code step shows complete code. ✅

**Type consistency:** `RetrievalFilter`, `Citation`, `_build_where`, `_document_to_citation`, `retrieve`, `Answer`, `get_vectorstore`, `format_sources` are used with identical signatures across tasks and tests. `Citation.score` defaults to `None`; `ask(..., filter=None)`; `retrieve(query, *, filter, k, settings)`. ✅
