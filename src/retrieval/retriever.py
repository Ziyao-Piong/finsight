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
