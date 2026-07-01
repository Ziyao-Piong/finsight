"""Phase 2 ingest: build a multi-filing, metadata-rich vector store from EDGAR.

This is the *write* half of the RAG loop, rebuilt for real corpora. Where Phase 1
fetched one hardcoded Apple 10-K and chunked it blindly, Phase 2 drives the
``src/ingest`` pipeline across several companies and years:

  for each (ticker, year):
      edgar.resolve_filing -> edgar.fetch_html      # discover + download from SEC
      parse.html_to_text   -> parse.segment_sections # clean + split into Items
      chunk.chunk_filing                              # section-aware chunks + metadata
  build_store(all_chunks, reset=...)                  # persist to Chroma

Every chunk lands with ``{company, ticker, form_type, fiscal_year, section, accession,
source_url, char_start, char_end, chunk_index}`` metadata — the foundation Phase 3's
filtered retrieval and citations build on.

Run it from the repo root with the venv active:

    python -m scripts.ingest                                   # defaults below
    python -m scripts.ingest --tickers AAPL,MSFT,NVDA --years 2023,2024 --reset

``--reset`` wipes this provider's collection first (use it on the first Phase 2 run to
clear Phase 1's metadata-free chunks). Without it, re-running upserts by chunk id, so
re-ingesting the same filings is idempotent. Switching ``LLM_PROVIDER`` changes the
embedding model and therefore the collection, so re-run after switching.
"""

from __future__ import annotations

import argparse

from src.config import get_settings
from src.ingest import store

# Phase 2 corpus: 3 companies x 2 fiscal years of annual reports. Enough to make
# cross-document comparison real (Phase 4) without blowing the time/compute budget.
DEFAULT_TICKERS = "AAPL,MSFT,NVDA"
DEFAULT_YEARS = "2023,2024"
DEFAULT_FORM = "10-K"


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest SEC filings into Chroma with section-aware chunks (Phase 2)."
    )
    parser.add_argument("--tickers", default=DEFAULT_TICKERS, help="Comma-separated tickers.")
    parser.add_argument("--years", default=DEFAULT_YEARS, help="Comma-separated fiscal years.")
    parser.add_argument("--form", default=DEFAULT_FORM, help="Filing form type (e.g. 10-K).")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete this provider's existing collection before ingesting.",
    )
    args = parser.parse_args()

    tickers = _split_csv(args.tickers)
    years = [int(y) for y in _split_csv(args.years)]

    settings = get_settings()
    print(f"Provider   : {settings.llm_provider}")
    print(f"Embeddings : {settings.embed_model_name}")
    print(f"Collection : {settings.collection_name}")
    print(f"Corpus     : {tickers} x {years} ({args.form})")
    print("-" * 70)

    all_chunks: list[dict] = []
    for ticker in tickers:
        for year in years:
            try:
                ref, sections, chunks = store.ingest_filing(ticker, year, args.form, settings)
            except Exception as exc:  # noqa: BLE001 — skip a bad filing, keep going
                print(f"  ! {ticker} FY{year}: skipped ({type(exc).__name__}: {exc})")
                continue

            section_names = ", ".join(s.label for s in sections) or "(none detected)"
            print(
                f"  + {ticker} FY{year}: {ref.accession} -> "
                f"{len(sections)} sections, {len(chunks)} chunks"
            )
            print(f"      sections: {section_names}")
            all_chunks.extend(chunks)

    print("-" * 70)
    if not all_chunks:
        print("No chunks produced — nothing to store. Check tickers/years/network.")
        return 1

    # Spot-check: show one chunk's metadata so the schema is visible at a glance.
    sample = all_chunks[0]["metadata"]
    print("Sample chunk metadata:")
    for key in sorted(sample):
        print(f"      {key}: {sample[key]!r}")

    print(f"\nEmbedding + persisting {len(all_chunks):,} chunks to Chroma...")
    persisted = store.build_store(all_chunks, settings, reset=args.reset)
    print(
        f"Done. {persisted._collection.count():,} chunks in "  # noqa: SLF001 — simple count
        f"'{settings.collection_name}' under {settings.chroma_dir}/."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
