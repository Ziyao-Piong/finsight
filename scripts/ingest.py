"""Phase 1 ingest: turn one 10-K into a searchable vector store.

This is the *write* half of the walking-skeleton RAG loop. It does the simplest
possible version of every step, on purpose:

  fetch one filing  ->  HTML to text  ->  naive fixed-size chunks  ->  embed  ->  Chroma

Everything here is throwaway-grade and gets replaced in later phases: Phase 2 turns
the single hardcoded fetch into a real EDGAR pipeline (many filings, rate limits,
section segmentation, metadata); for now we just need *something* in the store so the
retrieve->augment->generate loop has data to work with.

Run it from the repo root with the venv active:

    python -m scripts.ingest            # ingest (skips if already populated)
    python -m scripts.ingest --reset    # wipe this provider's collection and rebuild

Switching LLM_PROVIDER changes the embedding model and therefore the vector space,
so each provider gets its own Chroma collection — re-run this after switching.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Modern 10-K primary documents are Inline XBRL (XHTML/XML). Parsing them with the
# HTML parser still extracts clean text via get_text(), so silence the (correct but
# noisy) heads-up that this looks like XML. Phase 2's real parser handles structure.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from src.config import get_settings
from src.rag import get_vectorstore

# Apple Inc. FY2023 Form 10-K (fiscal year ended 2023-09-30), as filed on SEC EDGAR.
#   CIK 320193 · accession 0000320193-23-000106 · primary document aapl-20230930.htm
# Hardcoded on purpose for Phase 1 — Phase 2 generalizes filing discovery.
FILING_URL = (
    "https://www.sec.gov/Archives/edgar/data/320193/"
    "000032019323000106/aapl-20230930.htm"
)
# Cache the raw HTML here so re-running doesn't hammer EDGAR.
DATA_DIR = Path("data")
CACHE_PATH = DATA_DIR / "aapl-20230930.htm"


def fetch_filing(user_agent: str) -> str:
    """Return the filing's raw HTML, downloading once and caching it under data/."""
    if CACHE_PATH.exists():
        print(f"Using cached filing: {CACHE_PATH}")
        return CACHE_PATH.read_text(encoding="utf-8")

    print(f"Downloading filing from EDGAR: {FILING_URL}")
    # SEC requires a descriptive User-Agent with contact info, or it returns 403.
    resp = requests.get(FILING_URL, headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()

    DATA_DIR.mkdir(exist_ok=True)
    CACHE_PATH.write_text(resp.text, encoding="utf-8")
    print(f"Cached to {CACHE_PATH} ({len(resp.text):,} bytes)")
    return resp.text


def html_to_text(html: str) -> str:
    """Strip HTML to plain text with BeautifulSoup.

    Naive on purpose: we drop scripts/styles and collapse whitespace, but make no
    attempt to find sections (Item 1A, MD&A) or tables. That structural parsing is
    Phase 2's job — and the rough edges here are part of why Phase 2 exists.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    # Collapse runs of whitespace/newlines into single spaces/newlines.
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest one 10-K into Chroma (Phase 1).")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete this provider's existing collection before ingesting.",
    )
    args = parser.parse_args()

    settings = get_settings()
    print(f"Provider   : {settings.llm_provider}")
    print(f"Embeddings : {settings.embed_model_name}")
    print(f"Collection : {settings.collection_name}")
    print("-" * 60)

    store = get_vectorstore(settings)

    # Guard against accidentally embedding the same filing twice (which would skew
    # retrieval and waste embedding calls).
    try:
        existing = store._collection.count()  # noqa: SLF001 — simple count, no public API
    except Exception:
        existing = 0

    if existing and args.reset:
        print(f"--reset: deleting {existing} existing chunks...")
        store.delete_collection()
        store = get_vectorstore(settings)  # fresh handle after deletion
    elif existing:
        print(
            f"Collection already has {existing} chunks — nothing to do. "
            "Re-run with --reset to rebuild."
        )
        return 0

    html = fetch_filing(settings.edgar_user_agent)
    text = html_to_text(html)
    print(f"Extracted text: {len(text):,} characters")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks):,} chunks "
          f"(size={settings.chunk_size}, overlap={settings.chunk_overlap})")

    print("Embedding + persisting to Chroma (first run downloads the embed model)...")
    store.add_texts(chunks)
    print(f"Done. {store._collection.count():,} chunks stored in '{settings.collection_name}'.")  # noqa: SLF001
    print(f"Vector store persisted under: {settings.chroma_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
