# Phase 0 — Foundations & "hello, LLM"

> **Goal:** a clean project skeleton that can talk to an LLM through a *swappable*
> interface, so every later phase builds on a stable foundation.
>
> **Time budget:** 3–5 hours · **Status:** built, awaiting your run + review.

This guide is written for *you* to learn from. Read it top to bottom, then run the
smoke test at the end. By the time you finish you'll understand how the project
talks to a language model and why it's structured the way it is.

---

## 1. What we built and why

```
finsight/
├── requirements.txt        # Phase 0 dependencies only (kept minimal)
├── .env.example            # template for your secrets/config (copy to .env)
├── .gitignore              # keeps .env, .venv, caches out of git
├── agents/
│   └── plan.md             # the full living project plan (all 8 phases)
├── phase/
│   └── phase_0.md          # this file
├── scripts/
│   ├── __init__.py
│   └── hello_llm.py        # the runnable smoke test
└── src/
    ├── __init__.py
    ├── config.py           # all configuration, loaded from env / .env
    └── llm/
        ├── __init__.py
        └── factory.py      # returns a chat model / embeddings for the chosen provider
```

### File-by-file

- **`src/config.py`** — one typed object (`Settings`) holding every tunable value:
  which provider to use, model ids, API keys, paths. It reads from environment
  variables and your `.env` file. Nothing else in the codebase reads secrets
  directly; everyone calls `get_settings()`. This is the *single source of truth*.

- **`src/llm/factory.py`** — the provider switch. `get_chat_model()` returns a
  Claude or OpenAI chat model depending on `LLM_PROVIDER`; `get_embeddings()` does
  the same for embeddings. The rest of the app depends only on LangChain's abstract
  types (`BaseChatModel`, `Embeddings`), never on a specific vendor. **This is the
  one place that knows about vendors.**

- **`scripts/hello_llm.py`** — proves it all works: it sends a prompt to the
  configured model and **streams** the reply token-by-token to your terminal.

- **`.env.example`** — the template for the values `config.py` expects. You copy it
  to `.env` and fill in a key.

---

## 2. Concepts you're learning here

| Concept | What it means | Where you see it |
|---|---|---|
| **Chat model / LLM** | A model you send messages to and get a text reply from. | `factory.get_chat_model()` |
| **Prompt** | The text you send the model. Here, a plain string question. | `hello_llm.py` |
| **Token** | The unit a model reads/writes (roughly a word-piece). `LLM_MAX_TOKENS` caps how many it may generate. | `.env`, `config.py` |
| **Streaming** | Receiving the answer incrementally as it's generated, instead of waiting for the whole thing. Better UX and avoids timeouts on long outputs. | `chat.stream(...)` in `hello_llm.py` |
| **Embeddings** | A model that turns text into a vector of numbers so similar text lands near similar text. The backbone of search/RAG (you'll *use* this in Phase 1). | `factory.get_embeddings()` |
| **Provider abstraction** | Depending on an *interface* (`BaseChatModel`) instead of a concrete vendor class, so you can swap vendors via config. | `factory.py` |
| **12-factor config** | Configuration and secrets come from the environment, never hard-coded or committed. | `config.py`, `.env` |

A useful mental model for the whole project: **chat models reason and write;
embeddings find relevant text.** Phase 0 wires up the first. Phase 1 adds the second.

---

## 3. Setup (do this once)

You need **Python 3.10+** (3.11 recommended). Run everything from the repo root
(`finsight/` — the folder that contains `src/` and `scripts/`).

### 3a. Create and activate a virtual environment

A virtual environment keeps this project's packages isolated from the rest of your
system.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux (bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You'll know it worked when your prompt shows `(.venv)`.

### 3b. Install dependencies

```bash
pip install -r requirements.txt
```

### 3c. Create your `.env`

Copy the template and fill in **one** provider's key.

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**macOS / Linux:**
```bash
cp .env.example .env
```

Then open `.env` and either:
- keep `LLM_PROVIDER=anthropic` and paste your key into `ANTHROPIC_API_KEY=`
  (get one at https://console.anthropic.com/), **or**
- set `LLM_PROVIDER=openai` and paste your key into `OPENAI_API_KEY=`
  (get one at https://platform.openai.com/).

> You do **not** need a Voyage key for Phase 0 — that's only for embeddings, which
> start in Phase 1.

---

## 4. Run it

From the repo root, with the venv active:

```bash
python -m scripts.hello_llm
```

or pass your own question:

```bash
python -m scripts.hello_llm "Explain what a 10-K filing is in two sentences."
```

> Run it as a **module** (`python -m scripts.hello_llm`), not as a file path
> (`python scripts/hello_llm.py`). The `-m` form puts the repo root on Python's
> import path so `from src...` imports resolve.

### What you should see

```
Provider : anthropic
Model    : claude-sonnet-4-6
Prompt   : In one sentence, what is a financial 10-K filing?
------------------------------------------------------------
A 10-K is a comprehensive annual report that public companies file with the SEC...
```

The answer should appear **gradually**, a few words at a time — that's streaming
working. 🎉

---

## 5. Try the thing that makes this Phase matter

Open `.env`, change `LLM_PROVIDER` from `anthropic` to `openai` (or vice-versa),
make sure that provider's key is filled in, and run the **exact same command**
again. A different vendor answers — and you changed **zero lines of code**. That's
the provider abstraction paying off, and it's a point worth being able to explain
in an interview.

---

## 6. Design decisions worth understanding

- **Why a factory?** So "which vendor" is decided in one place. Later modules
  (retriever, agent, API) just call `get_chat_model()` and never import a vendor
  SDK. Swapping providers, or adding a third, touches one file.

- **Why is `temperature` left unset by default?** Newer Claude models
  (Opus 4.7 / 4.8, Fable 5) **reject** a `temperature` parameter and return an
  error. To keep model-swapping safe, `config.py` only passes `temperature` when
  you explicitly set `LLM_TEMPERATURE`. (`claude-sonnet-4-6` and `gpt-4o-mini` do
  accept it, so you can uncomment it for those.)

- **Why mirror keys into `os.environ`?** `pydantic-settings` loads `.env` into the
  `Settings` object, but LangChain integrations look for keys in real environment
  variables. `config._mirror_keys_to_env()` copies them across so it works whether
  your key lives in the shell or in `.env`.

- **Why minimal `requirements.txt`?** You only install what this phase uses. Heavy
  dependencies (vector store, FastAPI, parsing libs) arrive in the phase that needs
  them, so installs stay fast and the dependency list stays legible.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from the repo root using `python -m scripts.hello_llm` (not `python scripts/hello_llm.py`). |
| `[config error] ANTHROPIC_API_KEY is not set` | Create `.env` from `.env.example` and fill in the key for the selected provider. |
| `401` / authentication error | The key is wrong, expired, or for the other provider. Check `LLM_PROVIDER` matches the key you filled in. |
| `ModuleNotFoundError: langchain_anthropic` | `pip install -r requirements.txt` inside the activated venv. |
| Answer prints all at once, not gradually | Usually fine — some terminals buffer output. Streaming still happened. |

---

## 8. Phase 0 checklist

- [ ] `pip install -r requirements.txt` succeeds in a fresh venv.
- [ ] `.env` exists with a valid key for the selected provider.
- [ ] `python -m scripts.hello_llm` streams a sensible answer.
- [ ] Swapping `LLM_PROVIDER` and re-running works with no code changes.
- [ ] You can explain, in your own words: chat model, prompt, token, streaming,
      embeddings, and why the factory exists.

---

## 9. What's next — Phase 1 (preview)

Phase 1 builds a **walking-skeleton RAG**: ingest a single 10-K, chunk it, turn the
chunks into embeddings, store them in a local vector database (Chroma), then answer
a question by retrieving the most relevant chunks and feeding them to the chat model.
That's where `get_embeddings()` starts earning its keep and you see the full
retrieve → augment → generate loop end-to-end.

Tell me when you've run Phase 0 and we'll start Phase 1.
