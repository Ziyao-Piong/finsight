# Phase 2 — Production-shaped ingestion

> **Goal:** replace the toy single-filing ingest with a *real* pipeline — fetch several
> companies' 10-Ks from EDGAR, segment each into its proper sections, and chunk them with
> rich metadata, so every chunk knows *which company, year, and section* it came from.
>
> **Time budget:** 6–8 hours · **Status:** built, awaiting your run + review.
>
> Phase 1 proved the RAG loop on one document with no metadata. Phase 2 turns that toy
> into a small but real corpus and — crucially — gives every chunk the metadata that
> Phase 3 (filtered retrieval + citations) and Phase 4 (cross-document comparison) need.
> It's also the project's **first test-driven phase**.

---

## 1. What we built and why

```
finsight/
├── src/ingest/             # NEW package — the four-step pipeline
│   ├── edgar.py            # discover + fetch filings from SEC EDGAR by ticker/year
│   ├── parse.py            # HTML -> clean text, segment into labelled Items (1A, 7, 8…)
│   ├── chunk.py            # section-aware chunking + per-chunk metadata  ← the core
│   └── store.py            # orchestrate edgar->parse->chunk, persist to Chroma
├── scripts/ingest.py       # REWRITTEN: drives the pipeline over many (ticker, year) pairs
└── tests/                  # NEW — Phase 2 is test-driven
    ├── conftest.py         # a synthetic 10-K fixture (no network in tests)
    ├── test_parse.py       # section segmentation picks the body, not the table of contents
    └── test_chunk.py       # chunks never cross a section; metadata is complete + correct
```

The pipeline is four single-purpose stages wired together by `store.py`:

```
for each (ticker, year):
    edgar.resolve_filing → edgar.fetch_html       # SEC discovery + download (cached)
    parse.html_to_text   → parse.segment_sections  # clean text, split into Items
    chunk.chunk_filing                              # section-aware chunks + metadata
build_store(all_chunks, reset=…)                    # embed + persist to Chroma
```

Each chunk now carries:

```python
{ "company", "ticker", "form_type", "fiscal_year",
  "section", "accession", "source_url",
  "char_start", "char_end", "chunk_index" }
```

That metadata block is the whole point of the phase. Without it you can't filter
retrieval to "Apple's FY2023 risk factors," you can't cite a source, and you can't tell
the agent in Phase 4 which numbers belong to which filing.

---

## 2. Concepts you're learning here

| Concept | What it means | Where you see it |
|---|---|---|
| **Filing discovery** | Going from a human ticker ("AAPL") to a machine filing via SEC's ticker→CIK map and the submissions API. | `edgar.load_ticker_map`, `edgar.resolve_filing` |
| **Fiscal year ≠ filing date** | We key a filing's "year" on its *period of report* (`reportDate`), because fiscal calendars differ across companies. | `resolve_filing` (matches `report_date[:4]`) |
| **Rate limiting & identity** | Being a good API citizen: a descriptive `User-Agent` and ≤10 req/s, or SEC blocks you. | `edgar._get`, `_MIN_INTERVAL_S` |
| **Document structure** | Real filings have sections (Item 1A Risk Factors, Item 7 MD&A, Item 8 Financials); structure is information. | `parse.segment_sections` |
| **The TOC duplicate trap** | Every Item heading appears twice (table of contents + body). Naive matching grabs the wrong one. | "last occurrence wins" in `segment_sections` |
| **Section-aware chunking** | Splitting *within* a section so a chunk is never half-Risk-Factors, half-MD&A. | `chunk.chunk_filing` (splits each `Section` separately) |
| **Metadata schema design** | Deciding *what to record* on each chunk so later phases can filter, cite, and compare. | `chunk.REQUIRED_METADATA_KEYS` |
| **Deterministic ids / upsert** | Stable chunk ids mean re-ingesting overwrites instead of duplicating. | the `TICKER-YEAR-FORM-SECTION-INDEX` id |
| **Test-driven development** | Write the failing test first, watch it fail, then implement. | `test_parse.py` / `test_chunk.py`, written before the code |

The big mental shift from Phase 1: **ingestion is data engineering, not just text
loading.** The quality of everything downstream is capped by the structure and metadata
you capture here.

---

## 3. Why naive chunking had to go

Phase 1 stripped the whole 10-K to one string and cut it every 1000 characters. Two
things break on real filings:

1. **Chunks straddle topics.** A boundary can land mid-way between Risk Factors and the
   MD&A, producing a chunk that's half one topic and half another — noise for retrieval.
2. **Nothing is labelled.** Every chunk is anonymous text. You can't ask "only search
   Apple's FY2023 risk factors," and an answer can't say *where* it came from.

Phase 2 fixes both: we segment first, then chunk *inside* each section, and stamp every
chunk with its filing + section. Same `RecursiveCharacterTextSplitter` as Phase 1 — but
now it runs per section, and the output is labelled.

### The section-detection heuristic (and its honest limits)
A 10-K names sections like `Item 1A. Risk Factors`. The trap: that exact heading appears
in the table of contents *and* where the section really starts. We handle it with two
tricks (see the `parse.py` docstring):

- **Line-anchoring** — match `Item 1A` only at the start of a line, so cross-references
  buried in prose ("…as described in Item 8…") are ignored. That's why `html_to_text`
  keeps one line per source line.
- **Last occurrence wins** — of the (usually two) line-anchored matches, the body heading
  is the later one, because the whole TOC precedes the body.

This is a *text heuristic*, not a parser. A filing that repeats Items in an exhibit index
could mis-place a boundary. A production system would use the HTML structure (heading
tags/anchors) instead — a deliberate later upgrade, not a Phase 2 requirement. Capturing
"good enough structure now, measure later" is exactly the walking-skeleton philosophy.

---

## 4. Setup (deltas from Phase 1)

You already have the venv and embeddings configured. One new dependency:

```bash
pip install -r requirements.txt   # adds pytest (Phase 2 is test-driven)
```

No new *runtime* deps — `edgar.py`/`parse.py` reuse `requests` + BeautifulSoup, and the
chunker reuses the text splitter. Section parsing is hand-rolled on purpose (that's the
learning). Make sure `EDGAR_USER_AGENT` in `.env` is your real name + email; SEC needs it.

---

## 5. Run it

```bash
# 1. The tests come first in a TDD phase — they should all pass.
pytest -v

# 2. Build the corpus: 3 companies × 2 fiscal years of 10-Ks.
#    --reset clears Phase 1's metadata-free chunks on this first Phase 2 run.
python -m scripts.ingest --tickers AAPL,MSFT,NVDA --years 2023,2024 --reset

# 3. Ask across the richer corpus (the loop from Phase 1 still works unchanged).
python -m scripts.ask "What were Apple's main risk factors in FY2023?"
```

### What you should see

`ingest` reports each filing it found, the sections it detected, and a sample chunk's
metadata, then a final count:

```
Provider   : ollama
Embeddings : nomic-embed-text
Collection : finsight_ollama_nomic-embed-text
Corpus     : ['AAPL', 'MSFT', 'NVDA'] x [2023, 2024] (10-K)
----------------------------------------------------------------------
  + AAPL FY2023: 0000320193-23-000106 -> 16 sections, 612 chunks
      sections: Business, Risk Factors, …, MD&A, Market Risk, Financial Statements, …
  …
Sample chunk metadata:
      company: 'Apple Inc.'
      fiscal_year: 2023
      section: 'Business'
      ticker: 'AAPL'
      …
```

The **Phase 2 milestone**: a Chroma collection holding several filings whose every chunk
carries correct `{company, fiscal_year, section, …}` metadata. Spot-check it — that
metadata is what the next phase stands on.

> Note on providers: with `LLM_PROVIDER=ollama` (your current `.env`), all chunks embed
> locally, so the first full ingest of six filings takes a while. That's the local-OSS
> trade-off; a hosted embedder (OpenAI/Voyage) is faster.

---

## 6. Design decisions worth understanding

- **Why one module per stage?** `edgar` (I/O), `parse` (structure), `chunk` (pure
  transform), `store` (persistence). Each has a single job and a clean interface, so the
  pure transforms are trivially unit-testable and only `store.py` touches config.
- **Why is `chunk_filing` a pure function** (taking `chunk_size`/`chunk_overlap` instead
  of reading `Settings`)? Because pure functions are easy to test — the TDD skill's "hard
  to test = poor design" in action. `store.py` reads settings and passes the knobs in.
- **Why deterministic chunk ids?** `TICKER-YEAR-FORM-SECTION-INDEX` means re-ingesting
  upserts in place — idempotent re-runs without duplicating vectors.
- **Why match on `reportDate`, not filing date?** So companies line up by fiscal year for
  comparison even though their fiscal calendars differ (Apple ends Sept, NVIDIA ends Jan).
- **Why reuse `get_vectorstore` from `rag.py`?** So ingestion and querying can never
  disagree about collection name / embedding function / persist directory.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `403 Forbidden` from EDGAR | Set a real `EDGAR_USER_AGENT` (name + email) in `.env`. |
| `No 10-K found for TICKER with fiscal year YYYY` | The fiscal year is the *report-date* year (e.g. NVDA FY2024 ends Jan 2024). Try the adjacent year. |
| `Unknown ticker` | Check the symbol; the SEC map uses primary listings. Delete `data/company_tickers.json` to refresh it. |
| Few/odd sections detected | The text heuristic mis-read a messy filing — expected on some documents; the metadata is still attached, just labelled `Item N`. |
| First full ingest is slow | Embedding six filings (esp. locally via Ollama) is one-time per provider. |
| Want a clean rebuild | `python -m scripts.ingest --reset`. |

---

## 8. Phase 2 checklist

- [ ] `pip install -r requirements.txt` succeeds (pytest installed).
- [ ] `pytest -v` is green (parse + chunk tests).
- [ ] `python -m scripts.ingest --tickers AAPL,MSFT,NVDA --years 2023,2024 --reset` populates the collection.
- [ ] A spot-checked chunk has correct `{company, ticker, fiscal_year, section, char span}` metadata.
- [ ] `python -m scripts.ask "What were Apple's main risk factors in FY2023?"` returns a grounded answer.
- [ ] You can explain: ticker→CIK discovery, the TOC duplicate trap, why section-aware
      chunking beats naive chunking, and what each metadata field unlocks later.

---

## 9. What's next — Phase 3 (preview)

Phase 3 turns this metadata into accuracy and trust: a `retrieval/retriever.py` that
**filters** by `{company, fiscal_year, section}` before semantic search (so a question
about Apple doesn't surface Microsoft's chunks), and carries the source locators through
to the answer so it can render **expandable citations**. It's test-driven too
(`test_retriever.py` first). The schema you designed here is exactly what makes it possible.

Tell me when you've run Phase 2 and we'll start Phase 3.
