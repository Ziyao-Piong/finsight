# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FinSight is a **learning-oriented portfolio project** building an agentic RAG system over public SEC filings (10-K / 10-Q from EDGAR). The goal is to demonstrate AI/ML-Engineer skills (RAG, agentic tool-use, structured extraction, provider abstraction, and especially **evaluation rigor**) within a 40–60h budget — not to ship a polished product.

It is built in **iterative phases** (a "walking skeleton" model): Phase 1 stands up deliberately naive end-to-end RAG, and each later phase replaces one weak layer with a production-grade version *and measures the difference*. Because of this, prefer the simplest thing that completes the current phase over building ahead — later phases intentionally revisit and upgrade earlier code.

- `agents/plan.md` — the living master plan: full target architecture, all 8 phases (0–7), and end-to-end verification steps. **Read this to understand where any piece fits.**
- `phase/phase_N.md` — per-phase learning guide written for the human owner (concepts, setup, run, design rationale). Phase guides explain *why*, not just *what*.

**Current status: Phase 3 complete** (metadata-filtered retrieval + citations). Built so
far: Phase 0 skeleton, Phase 1 walking-skeleton RAG, Phase 2's `src/ingest/` pipeline, and
Phase 3's `src/retrieval/retriever.py` (`RetrievalFilter`, `_build_where`, `Citation`,
`retrieve`) with `ask()` returning an `Answer` (text + citations) and
`tests/test_retriever.py` / `tests/test_rag.py`. Still **not built yet**: `agent/`, `api/`,
`web/`, `eval/`.

## Commands

Run everything from the repo root with the venv active.

```bash
# One-time setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate            # macOS/Linux
pip install -r requirements.txt
cp .env.example .env                    # then fill in ONE provider's key

# Run the Phase 0 smoke test (streams an LLM reply)
python -m scripts.hello_llm
python -m scripts.hello_llm "Explain what a 10-K filing is in two sentences."
```

**Always run modules with `python -m <module>`, not `python path/to/file.py`.** The `-m` form puts the repo root on `sys.path` so `from src...` imports resolve; running a file directly will raise `ModuleNotFoundError: No module named 'src'`.

`requirements.txt` is kept **minimal on purpose** — it grows one phase at a time (currently through Phase 2: RAG + ingest + pytest). Add heavier libraries (langgraph, fastapi, uvicorn) in the phase that first needs them, not ahead of time.

## Architecture (current code)

Two hard rules define the structure; preserve them as the project grows:

1. **`src/config.py` is the single source of truth for configuration and secrets.** Nothing else reads `os.environ` directly for app config — callers use `get_settings()` (a cached `Settings`). The `LLM_PROVIDER` env var is the one switch that flips both chat model and embeddings between `anthropic` and `openai`.

2. **`src/llm/factory.py` is the only place that knows about a concrete vendor.** The rest of the app depends solely on LangChain's abstract `BaseChatModel` / `Embeddings`. Vendor SDKs are imported lazily *inside* each provider branch, so you only need the SDK for the provider you actually use. Adding a third provider or swapping vendors should touch only these two files.

### Provider/model gotchas (don't regress these)

- **Temperature is left unset by default.** Newer Claude models (Opus 4.7/4.8, Fable 5) **reject** a `temperature` parameter with a 400 error. `get_chat_model` only passes `temperature` when `LLM_TEMPERATURE` is explicitly set. Keep new model calls safe by default the same way.
- **API keys are mirrored into `os.environ`** by `config._mirror_keys_to_env()`. `pydantic-settings` loads `.env` into the `Settings` object, but LangChain integrations look for keys in real env vars (`ANTHROPIC_API_KEY`, etc.). The mirror copies them across and never overwrites a value already in the environment.
- **Anthropic has no first-party embeddings**, so the Anthropic path pairs with Voyage AI (`voyage-3`); the OpenAI path uses `text-embedding-3-small`.
- **Embeddings define vector-store identity** (relevant from Phase 1+): switching embedding provider means a different vector space and requires re-ingesting. Per the plan, key the Chroma collection name by provider to avoid silently mixing spaces.

## Conventions

- Phases 2–4 are **test-driven** per the plan (`test_chunk.py`, `test_retriever.py`, `test_tools.py` written first). Follow TDD when building those layers.
- Module docstrings in this codebase are intentionally teaching-oriented (they explain concepts for the human owner). Match that style — explain *why* a design choice was made, not just what the code does.
- Default chat model is `claude-sonnet-4-6` (good tool-use, cheaper than Opus); `claude-opus-4-8` is available via config.

## Contributing

- hide yourself when commit
- pytest is approved automatically