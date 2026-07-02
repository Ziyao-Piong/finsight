# FinSight — Financial Document Intelligence Agent

> **Status:** Phases 0–3 built (Foundations, walking-skeleton RAG, production-shaped
> ingestion, metadata-filtered retrieval + citations). Phases 4–7 not yet started.
> This is the living project plan. Phase-specific learning guides live in `phase/`.

## Context

This is a personal portfolio project whose goal is to build LLM/GenAI fluency and produce
a compelling artifact for **AI/ML Engineer** job applications. The build budget is **40–60
hours**. Success is measured less by product polish and more by how clearly it demonstrates
the skills those roles screen for: RAG, agentic tool-use, structured extraction, provider
abstraction, and — the key differentiator — **evaluation rigor**.

The chosen domain is **public SEC filings (10-K / 10-Q)** from EDGAR: free, legally clean,
text + tables dense enough to make retrieval genuinely hard, and instantly legible in a demo
("ask across Apple's last 3 annual reports and it cites the exact passage"). Scope is kept
tight — a handful of companies/years — to fit the time budget while remaining "scales to more."

The agent is **not a plain chatbot**. It exposes four composable capabilities: cited RAG Q&A,
cross-document comparison, structured extraction, and calculation tools. These compose into
one agentic design rather than four features.

## Goals

1. Cited RAG Q&A over SEC filings with source-passage citations.
2. Cross-document comparison (across companies / fiscal years) via multi-step agentic retrieval.
3. Schema-constrained structured extraction of key financials to JSON.
4. Calculation tools the agent calls (growth %, ratios, deltas) instead of hallucinating math.
5. Provider-swappable LLM + embeddings (Anthropic Claude ↔ OpenAI).
6. An automated evaluation harness (retrieval hit-rate + LLM-as-judge) showing before/after metrics.
7. FastAPI backend + minimal frontend for a shareable demo.

## Architecture

Each unit has one purpose, a defined interface, and is independently testable.

```
src/
  config.py            # pydantic-settings: provider choice, model ids, paths, API keys (from env)
  llm/
    factory.py         # get_chat_model() / get_embeddings() -> LangChain objects by config
  ingest/
    edgar.py           # fetch filings from SEC EDGAR (company facts + filing documents)
    parse.py           # HTML -> clean text, section segmentation (Item 1A, MD&A, financials)
    chunk.py           # section-aware chunking with metadata
    store.py           # build/persist Chroma collection
  retrieval/
    retriever.py       # metadata-filtered semantic search -> passages + source locators
  agent/
    tools/
      retrieve.py      # retrieve_filings tool
      extract.py       # extract_financials tool (schema-constrained output)
      calculate.py     # calculate tool (growth, ratio, delta)
    graph.py           # LangGraph tool-calling agent wiring the tools + system prompt
  api/
    app.py             # FastAPI: POST /ask (stream), POST /ingest, GET /health
web/
  index.html, app.js, style.css   # minimal chat UI: answer + expandable citations
eval/
  dataset.jsonl        # curated questions w/ expected sources / reference answers
  run_eval.py          # retrieval hit-rate + LLM-as-judge scoring; prints a metrics table
tests/
  test_chunk.py, test_retriever.py, test_tools.py, test_agent.py
```

### Provider abstraction (low-cost, high-signal)
LangChain already abstracts providers, so "swappable" is cheap:
- Chat: `langchain_anthropic.ChatAnthropic` (default `claude-sonnet-4-6`; `claude-opus-4-8` via config)
  vs `langchain_openai.ChatOpenAI`.
- Embeddings: `langchain_voyageai.VoyageAIEmbeddings` (`voyage-3`) on the Anthropic path
  (Anthropic has no first-party embeddings) vs `langchain_openai.OpenAIEmbeddings`
  (`text-embedding-3-small`) on the OpenAI path.
- `llm/factory.py` is the single switch point, driven by `config.py`.

> Note: the embedding model is part of the vector-store identity. Switching embedding
> providers requires re-ingesting (different vector space). Document this; key the Chroma
> collection name by provider to avoid silently mixing spaces.

**Open-source providers (added Phase 0.5).** Beyond the two frontier providers, `LLM_PROVIDER`
also accepts `groq` (hosted open-source — Llama 3.3 70B on Groq's free, OpenAI-compatible API;
embeds locally with HuggingFace `bge-small` since Groq has no embeddings endpoint) and `ollama`
(offline open-source running locally — `qwen2.5:7b` chat + `nomic-embed-text` embeddings). All
four live behind the same `factory.py` switch. This sets up a strong Phase 5 story: run the same
eval set across **frontier vs. hosted-OSS vs. local-OSS** and report the deltas. Caveat: reliable
multi-step tool-calling (Phase 4) is the weak spot for small local models — Groq is the dependable
agentic OSS path; Ollama is mainly for the offline/eval-comparison angle.

### Agent design
- LangGraph `create_react_agent`-style tool-calling loop (or a small custom `StateGraph`).
- System prompt instructs: always cite sources, use `calculate` for arithmetic, use
  `extract_financials` for structured numbers, never invent figures.
- Cross-doc comparison is emergent: the agent chains `retrieve_filings` (per company/year)
  → `extract_financials` → `calculate`.
- Stream tokens to the API for a responsive demo.

### Citations
Carry `{company, form_type, fiscal_year, section, chunk_id, char_span}` on each chunk's
metadata; retrieval returns these so the answer can render expandable source snippets.

## Build Phases (learning-oriented, iterative)

Phases follow an **iterative / walking-skeleton** model, not a build-each-layer-fully model.
Phase 1 stands up a deliberately *naive* end-to-end RAG so the whole loop runs early; each
later phase replaces one weak layer with a production-grade version and **measures the
difference**. Every phase ends in something runnable, so progress is always visible and each
GenAI concept is learned in the context of a working system rather than in isolation.

Each phase lists: **Goal · Build · Concepts you'll learn · Milestone · Verify.** Hour
estimates are rough and sum to the 40–60h budget.

### Phase 0 — Foundations & "hello, LLM" (3–5h)
- **Goal:** A clean project skeleton that can talk to an LLM through a swappable interface.
- **Build:** repo + `requirements.txt`, `.env.example`, `config.py` (`pydantic-settings`),
  `llm/factory.py` (`get_chat_model` / `get_embeddings`), `scripts/hello_llm.py`.
- **Learn:** Python project/dependency hygiene, secrets via env, LangChain chat-model basics,
  prompts, **token streaming**, the provider-abstraction pattern.
- **Milestone:** one script sends a prompt to Claude *and* OpenAI and streams the reply.
- **Verify:** `python -m scripts.hello_llm` streams a response for the configured provider.

### Phase 1 — Walking-skeleton RAG (5–7h)  ← most important learning milestone
- **Goal:** Prove the full RAG loop end-to-end on the simplest possible version.
- **Build:** ingest **one** 10-K with naive fixed-size chunking → embed → Chroma → a plain
  retrieve-then-stuff-into-prompt Q&A function (no agent, no metadata yet).
- **Learn:** what embeddings are, vector similarity, the retrieve→augment→generate loop,
  context stuffing, and a first feel for why chunking choices matter.
- **Milestone:** ask a question about that one filing and get a grounded answer.
- **Verify:** CLI `ask("What were the main risk factors?")` returns an answer drawn from the doc.

### Phase 2 — Production-shaped ingestion (6–8h)
- **Goal:** Replace the toy ingest with a real, multi-document pipeline.
- **Build:** `ingest/edgar.py` (EDGAR fetch, correct `User-Agent`, rate limits) → `parse.py`
  (HTML→text, section segmentation: Item 1A, MD&A, financial statements) → `chunk.py`
  (section-aware chunking + metadata) → `store.py`. Ingest ~3 companies × 2 years.
- **Learn:** data engineering for RAG, document structure, **metadata schema design**, why
  naive chunking breaks on real filings. (TDD: `test_chunk.py`.)
- **Milestone:** a Chroma corpus of several filings with rich, queryable metadata.
- **Verify:** collection populated; spot-check chunk metadata (`company`, `fiscal_year`, `section`).

### Phase 3 — Better retrieval & citations (4–6h)
- **Goal:** Make retrieval accurate and answers verifiable.
- **Build:** `retrieval/retriever.py` with metadata filtering; carry source locators through
  to the answer for expandable citations.
- **Learn:** retrieval quality, metadata filtering, grounding & citation rendering.
- **Milestone:** answers cite exact, correctly-attributed passages.
- **Verify:** `pytest tests/test_retriever.py` — relevant passages with correct metadata (TDD).
- **Status: ✅ Built.** `src/retrieval/retriever.py` (`RetrievalFilter`, `_build_where`,
  `Citation`, `retrieve`); `ask()` returns an `Answer` with citations; `tests/test_retriever.py`.

### Phase 4 — Agentic tools (8–12h)  ← the AI/ML-Engineer centerpiece
- **Goal:** Turn the pipeline into an agent that reasons across multiple steps and documents.
- **Build:** three tools (`retrieve_filings`, `extract_financials` with schema-constrained
  output, `calculate`), then `agent/graph.py` — a LangGraph tool-calling loop with a system
  prompt enforcing "cite sources, use tools for math/numbers, never invent figures."
- **Learn:** tool/function calling, agent loops, **structured/JSON-constrained output**,
  multi-step reasoning; cross-doc comparison emerges from chaining the tools.
- **Milestone:** "How did NVIDIA's revenue change FY2023→FY2024?" → cites both filings and
  shows the computed growth %.
- **Verify:** `pytest tests/test_tools.py` (TDD each tool first) + a manual agent transcript.

### Phase 5 — Evaluation harness (6–8h)  ← the interview differentiator
- **Goal:** Be able to prove the system works and measure when it improves.
- **Build:** `eval/dataset.jsonl` (~15–25 curated Qs w/ expected sources/reference answers) +
  `eval/run_eval.py` computing **retrieval hit-rate** and **LLM-as-judge** answer quality.
- **Learn:** eval methodology, the metric vocabulary interviews probe, reading/acting on scores.
- **Milestone:** baseline metrics, then **one deliberate improvement** (e.g. chunk size or
  metadata filtering) with a recorded **before/after delta** — the headline story.
- **Verify:** `python eval/run_eval.py` prints a metrics table; run twice across the change.

### Phase 6 — Serve it: API + frontend (6–8h)
- **Goal:** Wrap the engine in a shareable demo.
- **Build:** `api/app.py` (FastAPI: `POST /ask` streaming, `POST /ingest`, `GET /health`) +
  minimal `web/` chat UI rendering answer + expandable citations.
- **Learn:** serving LLM apps, **streaming over HTTP (SSE)**, API design, request/response shapes.
- **Milestone:** type a question in the browser, watch a cited answer stream in.
- **Verify:** `uvicorn src.api.app:app` → `POST /ask` streams; frontend renders it.

### Phase 7 — Polish & portfolio (3–5h)
- **Goal:** Make it land in 60 seconds for a reviewer.
- **Build:** README with architecture diagram, demo GIF, the eval metrics table, setup steps,
  and a "what I'd do next" section; tidy provider-swap docs.
- **Learn:** communicating technical work — often as important as the code in a job search.
- **Milestone:** a repo a recruiter/engineer can understand and run quickly.
- **Verify:** a fresh clone + README steps reaches a working demo.

**Provider-swap checkpoint (any time after Phase 4):** flip `LLM_PROVIDER` in `.env`, re-ingest
(embeddings differ → different vector space), re-run a query to confirm the abstraction holds.

## Key Reuse / Libraries
- `langchain`, `langgraph`, `langchain-anthropic`, `langchain-openai`, `langchain-voyageai`
- `langchain-chroma` (or `chromadb`) for the vector store
- SEC EDGAR access via the official submissions/full-text endpoints (respect the required
  `User-Agent` header and rate limits) + `beautifulsoup4`/`lxml` for parsing
- `fastapi`, `uvicorn`, `pydantic-settings`
- `pytest` for tests and eval assertions

## End-to-end verification (once all phases land)
1. **Ingest**: `python -m src.ingest.run --tickers AAPL,MSFT,NVDA --years 2023,2024`.
2. **Retrieval**: `pytest tests/test_retriever.py`.
3. **Tools**: `pytest tests/test_tools.py`.
4. **Agent**: "How did NVIDIA's revenue change FY2023→FY2024?" → cites both filings + growth %.
5. **API**: `uvicorn src.api.app:app` → `POST /ask` streams a cited answer.
6. **Eval**: `python eval/run_eval.py` → metrics table; run twice across a deliberate change.
7. **Provider swap**: flip `LLM_PROVIDER`, re-ingest, re-run a query.

## Out of Scope (YAGNI)
Auth/multi-user, production deployment/hosting, real-time filing updates, hundreds of filings,
fine-tuning. Note these in the README as future work.
