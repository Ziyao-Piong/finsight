"""Orchestrate the ingest pipeline and persist chunks to Chroma.

This is the seam that ties the three single-purpose modules together — fetch
(:mod:`edgar`) -> segment (:mod:`parse`) -> chunk (:mod:`chunk`) — and writes the
result into the vector store. It is also the one place in the ingest package that reads
:class:`~src.config.Settings`, so the chunker stays a pure function and the vector-store
wiring stays shared with the query side.

Two deliberate reuse choices keep Phase 2 consistent with the rest of the app:

* We persist through :func:`src.rag.get_vectorstore`, the *same* handle the query path
  uses, so ingestion and retrieval can never disagree about the collection name,
  embedding function, or persist directory.
* We pass each chunk's deterministic id to ``add_texts``. Chroma treats matching ids as
  an upsert, so re-running ingest overwrites a filing's chunks in place instead of
  duplicating them — re-ingest is idempotent without a full ``--reset``.
"""

from __future__ import annotations

import time

from src.config import Settings, get_settings
from src.ingest import chunk as chunk_mod
from src.ingest import edgar, parse
from src.ingest.edgar import FilingRef
from src.ingest.parse import Section
from src.rag import get_vectorstore

# Embedding a few thousand chunks in one call is fragile against a local embedder: the
# Ollama model runner can drop the connection mid-batch, losing the whole ingest. We
# embed in modest batches and retry a batch a few times (the runner restarts on the next
# request), so a transient hiccup costs one batch, not the run. Hosted embedders don't
# need this but it's harmless for them.
_EMBED_BATCH_SIZE = 100
_EMBED_MAX_RETRIES = 3
_EMBED_RETRY_WAIT_S = 2.0


def ingest_filing(
    ticker: str, fiscal_year: int, form: str, settings: Settings | None = None
) -> tuple[FilingRef, list[Section], list[dict]]:
    """Run fetch -> parse -> chunk for one filing.

    Returns the resolved :class:`FilingRef`, its sections, and the chunk dicts (with
    metadata + ids). The script layer uses the ref/sections for human-readable summaries
    and hands the chunks to :func:`build_store`.
    """
    settings = settings or get_settings()
    ref = edgar.resolve_filing(ticker, fiscal_year, form, settings)
    html = edgar.fetch_html(ref, settings)
    text = parse.html_to_text(html)
    sections = parse.segment_sections(text)
    chunks = chunk_mod.chunk_filing(
        sections,
        company=ref.company,
        ticker=ref.ticker,
        form_type=ref.form,
        fiscal_year=ref.fiscal_year,
        accession=ref.accession,
        source_url=ref.source_url,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return ref, sections, chunks


def build_store(chunks: list[dict], settings: Settings | None = None, reset: bool = False):
    """Write chunk dicts (text + metadata + id) to the provider's Chroma collection.

    Args:
        chunks: ``{"text", "metadata", "id"}`` dicts from :func:`src.ingest.chunk.chunk_filing`.
        settings: Optional settings override (defaults to the shared cached settings).
        reset: When True, drop the collection first so Phase 2's structured chunks don't
            mix with any older (e.g. Phase 1 metadata-free) chunks in the same space.

    Returns:
        The Chroma store handle (already persisted on disk).
    """
    settings = settings or get_settings()
    store = get_vectorstore(settings)

    if reset:
        store.delete_collection()
        store = get_vectorstore(settings)  # fresh handle after deletion

    if not chunks:
        return store

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [c["id"] for c in chunks]

    total = len(texts)
    for start in range(0, total, _EMBED_BATCH_SIZE):
        end = min(start + _EMBED_BATCH_SIZE, total)
        _add_batch_with_retry(
            store,
            texts[start:end],
            metadatas[start:end],
            ids[start:end],
        )
        print(f"  embedded {end:,}/{total:,} chunks", flush=True)
    return store


def _add_batch_with_retry(store, texts: list[str], metadatas: list[dict], ids: list[str]):
    """Embed + add one batch, retrying on transient embedder failures.

    Deterministic ids make this safe to retry: a partially-applied batch upserts the
    same ids on the next attempt rather than duplicating.
    """
    for attempt in range(1, _EMBED_MAX_RETRIES + 1):
        try:
            store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            return
        except Exception as exc:  # noqa: BLE001 — embedder/transport errors are transient
            if attempt == _EMBED_MAX_RETRIES:
                raise
            print(
                f"  ! embed batch failed ({type(exc).__name__}); "
                f"retry {attempt}/{_EMBED_MAX_RETRIES - 1} after {_EMBED_RETRY_WAIT_S}s",
                flush=True,
            )
            time.sleep(_EMBED_RETRY_WAIT_S)
