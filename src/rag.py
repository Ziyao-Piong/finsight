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

from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import Settings, get_settings
from src.llm.factory import get_chat_model
from src.retrieval.retriever import Citation, RetrievalFilter, get_vectorstore, retrieve

_SYSTEM_PROMPT = (
    "You are FinSight, a financial-filings assistant. Answer the user's question "
    "using ONLY the context passages provided below, which are excerpts from an SEC "
    "filing. If the answer is not contained in the context, say you don't have enough "
    "information from the filing to answer — do not use outside knowledge or guess. "
    "Be concise and factual."
)


@dataclass(frozen=True)
class Answer:
    """A grounded answer plus the passages it was drawn from.

    ``text`` is the model's answer; ``citations`` are the retrieved passages (in the same
    order they were numbered in the context) so a UI can render expandable sources.
    """

    text: str
    citations: list[Citation]


def _format_context(citations: list[Citation]) -> str:
    """Number each passage ``[i]`` so the model can attribute claims to a source.

    The numbering matches the order of ``citations`` on the returned :class:`Answer`, so a
    later UI can line prose up with the expandable source it came from.
    """
    return "\n\n".join(
        f"[{i}] {c.company} {c.form_type} FY{c.fiscal_year} — {c.section}\n{c.snippet}"
        for i, c in enumerate(citations, start=1)
    )


def ask(
    question: str,
    settings: Settings | None = None,
    k: int | None = None,
    filter: RetrievalFilter | None = None,  # noqa: A002
    stream: bool = True,
) -> Answer:
    """Answer ``question`` from the ingested filings and return an :class:`Answer`.

    Args:
        question: The user's natural-language question.
        settings: Optional settings override (defaults to the shared cached settings).
        k: How many chunks to retrieve. Defaults to ``settings.retrieval_k``.
        filter: Optional metadata constraints (ticker / fiscal year / section / form).
            The Phase 4 agent will supply these; the CLI leaves it ``None``.
        stream: When True (default), print tokens to stdout as they arrive.

    Returns:
        An :class:`Answer` with the answer text and the citations it was grounded in.
    """
    settings = settings or get_settings()
    k = k or settings.retrieval_k

    # --- Retrieve -------------------------------------------------------------
    citations = retrieve(question, filter=filter, k=k, settings=settings)

    if not citations:
        msg = (
            "No chunks found in the vector store. Have you run "
            "`python -m scripts.ingest` for this provider yet?"
        )
        if stream:
            print(msg)
        return Answer(text=msg, citations=[])

    # --- Augment --------------------------------------------------------------
    context = _format_context(citations)
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

    return Answer(text="".join(parts), citations=citations)
