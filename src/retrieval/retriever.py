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

from dataclasses import dataclass

from src.config import Settings, get_settings


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
