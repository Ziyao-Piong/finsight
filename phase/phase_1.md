# Phase 1 — Walking-skeleton RAG

> **Goal:** prove the *full* retrieval-augmented-generation loop end-to-end on the
> simplest possible version — ingest one 10-K, then ask it questions and get answers
> grounded in the document.
>
> **Time budget:** 5–7 hours · **Status:** built, awaiting your run + review.
>
> This is the most important *learning* milestone in the project. Phase 0 taught the
> app to talk to a model; Phase 1 teaches it to talk about *your documents*.

---

## 1. What we built and why

```
finsight/
├── data/                   # cached downloaded filings (gitignored)
├── scripts/
│   ├── ingest.py           # WRITE half: fetch -> text -> chunk -> embed -> Chroma
│   └── ask.py              # READ half: question -> grounded, streamed answer
└── src/
    ├── config.py           # + chunk/overlap/k knobs, EDGAR user-agent, collection name
    └── rag.py              # the retrieve -> augment -> generate loop
```

The whole phase is two halves of one loop:

- **`scripts/ingest.py` (write once).** Downloads Apple's FY2023 10-K from SEC EDGAR
  (cached under `data/`), strips the HTML to text with BeautifulSoup, splits it into
  fixed-size chunks, turns each chunk into an embedding, and stores them in a local
  Chroma database. You run this once per provider.

- **`src/rag.py` (the loop).** `ask(question)` embeds your question, finds the most
  similar chunks in Chroma, pastes them into the prompt as "context," and streams the
  model's answer — with a system prompt that says *answer only from the context*.

- **`scripts/ask.py` (read any time).** A thin CLI over `ask()` so you can type
  questions at the terminal.

Everything here is **deliberately naive**. No metadata, no citations, no section
awareness, one document. That's the point of a walking skeleton: get the entire loop
running first, then let later phases replace each weak layer *and measure the gain*.

---

## 2. Concepts you're learning here

| Concept | What it means | Where you see it |
|---|---|---|
| **Embedding** | A model that turns text into a vector of numbers, positioned so similar meanings land near each other. | `get_embeddings()`, used inside `get_vectorstore()` |
| **Vector store** | A database of those vectors that can quickly find the nearest ones to a query vector. Here: Chroma, persisted on disk. | `src/rag.py:get_vectorstore` |
| **Vector similarity / semantic search** | Finding relevant text by *meaning* (vector closeness), not keyword matching. "risk factors" can match a passage that never says those exact words. | `store.similarity_search(question, k)` |
| **Chunking** | Splitting a long document into pieces small enough to embed and to fit in the prompt. We use naive fixed-size chunks with overlap. | `RecursiveCharacterTextSplitter` in `ingest.py` |
| **Overlap** | Repeating a tail of one chunk at the start of the next, so an answer straddling a boundary isn't cut in half. | `CHUNK_OVERLAP` |
| **Retrieval `k`** | How many chunks to pull back and stuff into the prompt. Bigger `k` = more context but more noise and tokens. | `RETRIEVAL_K`, `ask(..., k=)` |
| **Augment / context stuffing** | Inserting retrieved passages into the prompt so the model answers from them instead of memory. | `_format_context` + the `HumanMessage` in `rag.py` |
| **Grounding** | Constraining the model to the supplied passages and having it admit when the answer isn't there. | `_SYSTEM_PROMPT` in `rag.py` |

The mental model from Phase 0 now completes: **embeddings find relevant text; the
chat model reasons over it and writes the answer.** RAG is just wiring those two together.

---

## 3. Setup (deltas from Phase 0)

You already have the venv from Phase 0. Two things to do:

### 3a. Install the new dependencies

```bash
pip install -r requirements.txt
```

This adds the vector store (`langchain-chroma`), the text splitter, BeautifulSoup +
`lxml` for HTML, and `requests`. The first ingest run may also download an embedding
model (Voyage/OpenAI hit an API; Groq's HuggingFace model downloads ~130 MB locally).

### 3b. Make sure embeddings are configured for your provider

Phase 0 only needed a chat key. Phase 1 also **embeds**, and that's a separate model:

| `LLM_PROVIDER` | Embeddings used | What you need |
|---|---|---|
| `anthropic` (default) | Voyage AI `voyage-3` | `VOYAGE_API_KEY` in `.env` |
| `openai` | `text-embedding-3-small` | `OPENAI_API_KEY` (same key as chat) |
| `groq` | local HuggingFace `bge-small` | nothing (downloads once) |
| `ollama` | `nomic-embed-text` | `ollama pull nomic-embed-text` |

Also set `EDGAR_USER_AGENT` in `.env` to your own name + email — SEC requires it.

---

## 4. Run it

From the repo root, with the venv active:

```bash
# 1. Build the index (downloads the filing once, then embeds it)
python -m scripts.ingest

# 2. Ask the milestone question
python -m scripts.ask "What were the main risk factors?"
```

### What you should see

`ingest` prints its progress and a final chunk count:

```
Provider   : anthropic
Embeddings : voyage-3
Collection : finsight_anthropic_voyage-3
------------------------------------------------------------
Downloading filing from EDGAR: https://www.sec.gov/Archives/.../aapl-20230930.htm
Extracted text: 412,533 characters
Split into 480 chunks (size=1000, overlap=150)
Embedding + persisting to Chroma ...
Done. 480 chunks stored in 'finsight_anthropic_voyage-3'.
```

`ask` streams a grounded answer drawn from the filing — describing Apple's actual
risk factors (supply concentration, competition, macro/FX, legal/regulatory, etc.),
not generic boilerplate. That's the **Phase 1 milestone**. 🎉

Try a couple more to feel the loop:

```bash
python -m scripts.ask "What products does the company sell?"
python -m scripts.ask "Who is the CEO of Tesla?"     # out of scope -> it should decline
```

The last one isn't in an Apple 10-K, so a well-grounded system says it doesn't have
that information — proving the answer really is coming from the document.

---

## 5. Try the thing that makes this phase matter

Open `src/config.py` (or `.env`) and change `CHUNK_SIZE` — say from `1000` to `300`.
Then rebuild and re-ask:

```bash
python -m scripts.ingest --reset
python -m scripts.ask "What were the main risk factors?"
```

Smaller chunks = more, narrower passages: retrieval gets more precise but each passage
carries less surrounding context, so answers can feel fragmented. Bigger chunks do the
opposite. There's no universally right number — and *that* is the lesson. You're feeling,
firsthand, why chunking is a real design decision. Phase 5's eval harness will let you
put a **number** on which choice is better instead of eyeballing it.

---

## 6. Design decisions worth understanding

- **Why one hardcoded filing instead of a real fetcher?** Phase 1 only needs *data in
  the store* to prove the loop. Building EDGAR discovery, rate limiting, and section
  parsing now would be building ahead — that's Phase 2, which replaces `ingest.py`'s
  fetch step wholesale.

- **Why naive fixed-size chunking?** It ignores the document's structure (it'll happily
  cut a sentence or a table in half). It's the baseline we *measure against* later. Real
  filings have sections (Item 1A Risk Factors, MD&A) that section-aware chunking in
  Phase 2 will respect.

- **Why is the collection name keyed by provider?** An embedding model defines a vector
  space; vectors from `voyage-3` and OpenAI aren't comparable. `Settings.collection_name`
  bakes the provider + embed model into the name so the four providers can never
  silently mix vectors in one collection.

- **Why must I re-ingest after switching providers?** Same reason: a different provider
  means a different embedding model means a different (empty, for that provider) vector
  space. `ingest.py` will just build a new collection alongside the old one.

- **Why does `ask` live in `src/rag.py` and not in the script?** So the API and the
  agent in later phases can import and reuse the exact same loop. Scripts stay thin.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `403 Forbidden` from EDGAR | Set a real `EDGAR_USER_AGENT` (name + email) in `.env`. SEC blocks generic/empty agents. |
| `ModuleNotFoundError: langchain_chroma` (or `bs4`) | `pip install -r requirements.txt` inside the activated venv. |
| `ask` says it has no information / returns nothing | You haven't ingested for this provider. Run `python -m scripts.ingest`. Each provider has its own collection. |
| Switched provider, answers look wrong/empty | Re-run `python -m scripts.ingest` — the new provider's collection starts empty. |
| Want to rebuild after changing chunk settings | `python -m scripts.ingest --reset`. |
| First ingest is slow | The embedding model is downloading (HuggingFace) or you're embedding ~hundreds of chunks via API. One-time per provider. |

---

## 8. Phase 1 checklist

- [ ] `pip install -r requirements.txt` succeeds.
- [ ] Embeddings configured for your `LLM_PROVIDER` (Voyage/OpenAI key, or local model).
- [ ] `python -m scripts.ingest` reports a non-trivial chunk count and persists `.chroma/`.
- [ ] `python -m scripts.ask "What were the main risk factors?"` streams a grounded answer.
- [ ] An out-of-scope question is declined (proves grounding).
- [ ] You can explain, in your own words: embedding, vector similarity, chunking + overlap,
      retrieval `k`, context stuffing, and why switching providers needs a re-ingest.

---

## 9. What's next — Phase 2 (preview)

Phase 2 replaces the toy ingest with a **real, multi-document pipeline**: a proper
EDGAR fetcher (`ingest/edgar.py`) with the correct headers and rate limits, HTML→text
with **section segmentation** (Item 1A, MD&A, financial statements), **section-aware
chunking with metadata** (`chunk.py`), and a store builder (`store.py`) — ingesting a
few companies × a couple of years. That's where chunks gain the `{company, fiscal_year,
section, ...}` metadata that Phase 3's filtered retrieval and citations depend on. It's
also the first **test-driven** phase (`test_chunk.py` written first).

Tell me when you've run Phase 1 and we'll start Phase 2.
