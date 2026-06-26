"""The walking-skeleton RAG loop: retrieve -> augment -> generate.

This is the heart of Phase 1 and the smallest thing that proves the whole idea of
retrieval-augmented generation end to end. The flow is:

  1. **Retrieve** — embed the question and ask the vector store for the chunks whose
     embeddings sit closest to it (semantic similarity, not keyword matching).
  2. **Augment** — paste those chunks into the prompt as "context" and instruct the
     model to answer *only* from them. This is what grounds the answer in the source
     document instead of the model's parametric memory.
  3. **Generate** — stream the model's answer token-by-token.

It is deliberately naive: no metadata, no citations, no query rewriting, a single
collection. Those are exactly the weak layers Phases 2–4 replace and *measure*. The
job here is to make the loop real and visible, not to make it good yet.

Like the rest of the app, this module is provider-agnostic: it only touches
``get_chat_model`` / ``get_embeddings`` from the factory, so the same code runs on
Anthropic, OpenAI, Groq, or Ollama.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import Settings, get_settings
from src.llm.factory import get_chat_model, get_embeddings

# Instruction that keeps the model honest: answer from the retrieved passages, and
# admit when they don't contain the answer instead of inventing one. Grounding +
# refusal is the behaviour every later phase builds on.
_SYSTEM_PROMPT = (
    "You are FinSight, a financial-filings assistant. Answer the user's question "
    "using ONLY the context passages provided below, which are excerpts from an SEC "
    "filing. If the answer is not contained in the context, say you don't have enough "
    "information from the filing to answer — do not use outside knowledge or guess. "
    "Be concise and factual."
)


def get_vectorstore(settings: Settings | None = None):
    """Return the persistent Chroma vector store for the current provider.

    Both ingestion (writing chunks) and querying (reading them) go through this one
    function so they always agree on three things: the persist directory, the
    embedding function, and the collection name. The collection name is keyed by
    provider (see ``Settings.collection_name``) because each provider's embeddings
    live in their own, incomparable vector space.
    """
    settings = settings or get_settings()
    # Imported lazily so importing this module doesn't require chromadb until you
    # actually build or query an index (mirrors the factory's lazy-import style).
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings),
        persist_directory=settings.chroma_dir,
    )


def _format_context(docs) -> str:
    """Join retrieved chunks into a single context block, numbered for readability."""
    return "\n\n".join(
        f"[Passage {i}]\n{doc.page_content}" for i, doc in enumerate(docs, start=1)
    )


def ask(
    question: str,
    settings: Settings | None = None,
    k: int | None = None,
    stream: bool = True,
) -> str:
    """Answer ``question`` from the ingested filing and return the answer text.

    Args:
        question: The user's natural-language question.
        settings: Optional settings override (defaults to the shared cached settings).
        k: How many chunks to retrieve. Defaults to ``settings.retrieval_k``.
        stream: When True (default), print tokens to stdout as they arrive so you can
            watch the answer build — the same streaming UX as the Phase 0 smoke test.

    Returns:
        The full answer as a string (also printed live when ``stream`` is True).
    """
    settings = settings or get_settings()
    k = k or settings.retrieval_k

    # --- Retrieve -------------------------------------------------------------
    store = get_vectorstore(settings)
    docs = store.similarity_search(question, k=k)

    if not docs:
        msg = (
            "No chunks found in the vector store. Have you run "
            "`python -m scripts.ingest` for this provider yet?"
        )
        if stream:
            print(msg)
        return msg

    # --- Augment --------------------------------------------------------------
    context = _format_context(docs)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]

    # --- Generate -------------------------------------------------------------
    chat = get_chat_model(settings)
    parts: list[str] = []
    for chunk in chat.stream(messages):
        text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
        parts.append(text)
        if stream:
            print(text, end="", flush=True)
    if stream:
        print()  # final newline after the streamed answer

    return "".join(parts)
