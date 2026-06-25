"""Provider-agnostic factory for chat models and embeddings.

This module is the SINGLE place in the codebase that knows which concrete LLM
provider we use. Everything else just calls :func:`get_chat_model` /
:func:`get_embeddings` and stays provider-agnostic — so swapping Anthropic <->
OpenAI is a one-line change in ``.env`` (``LLM_PROVIDER``), not a code change.

This "depend on an interface, not a concrete class" pattern is the same idea as
dependency injection: the rest of the app depends on LangChain's abstract
``BaseChatModel`` / ``Embeddings`` types, and this factory decides which
implementation to hand back.

Imports of the provider packages are done lazily INSIDE each branch so you only
need the SDK for the provider you're actually using installed/working.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import Settings, get_settings


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Return a streaming chat model for the configured provider.

    Args:
        settings: Optional settings override (handy in tests). Defaults to the
            shared cached settings.

    Returns:
        A LangChain ``BaseChatModel`` ready to ``.invoke(...)`` or ``.stream(...)``.
    """
    settings = settings or get_settings()
    provider = settings.llm_provider

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs: dict = {
            "model": settings.anthropic_chat_model,
            "max_tokens": settings.llm_max_tokens,
            "streaming": True,
        }
        # Only pass temperature when explicitly set — newer Claude models reject it.
        if settings.llm_temperature is not None:
            kwargs["temperature"] = settings.llm_temperature
        return ChatAnthropic(**kwargs)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": settings.openai_chat_model,
            "max_tokens": settings.llm_max_tokens,
            "streaming": True,
        }
        if settings.llm_temperature is not None:
            kwargs["temperature"] = settings.llm_temperature
        return ChatOpenAI(**kwargs)

    if provider == "groq":
        # Groq's API is OpenAI-COMPATIBLE, so we reuse ChatOpenAI and just point it
        # at Groq's base_url with the Groq key. This keeps the hosted-OSS branch
        # vendor-generic — OpenRouter/Together would only need different env values.
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": settings.groq_chat_model,
            "base_url": settings.groq_base_url,
            "api_key": settings.groq_api_key,
            "max_tokens": settings.llm_max_tokens,
            "streaming": True,
        }
        if settings.llm_temperature is not None:
            kwargs["temperature"] = settings.llm_temperature
        return ChatOpenAI(**kwargs)

    if provider == "ollama":
        # Fully local: talks to an Ollama server (default http://localhost:11434).
        # Ollama caps generation length via `num_predict` (its name for max_tokens),
        # and ChatOllama.bind_tools() provides the tool-calling used from Phase 4.
        from langchain_ollama import ChatOllama

        kwargs = {
            "model": settings.ollama_chat_model,
            "base_url": settings.ollama_base_url,
            "num_predict": settings.llm_max_tokens,
            "streaming": True,
        }
        if settings.llm_temperature is not None:
            kwargs["temperature"] = settings.llm_temperature
        return ChatOllama(**kwargs)

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r} "
        "(expected 'anthropic', 'openai', 'groq', or 'ollama')"
    )


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Return an embeddings model for the configured provider.

    Not exercised in Phase 0 (which only does chat), but defined now so the
    provider switch lives in exactly one place. Note that two providers have no
    embeddings API of their own: Anthropic pairs with Voyage AI, and Groq pairs
    with a local HuggingFace (sentence-transformers) model. Ollama serves its
    own embedding model locally.

    Args:
        settings: Optional settings override. Defaults to shared cached settings.

    Returns:
        A LangChain ``Embeddings`` implementation.
    """
    settings = settings or get_settings()
    provider = settings.llm_provider

    if provider == "anthropic":
        from langchain_voyageai import VoyageAIEmbeddings

        return VoyageAIEmbeddings(model=settings.voyage_embed_model)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.openai_embed_model)

    if provider == "groq":
        # Groq has no embeddings endpoint, so the hosted-OSS path embeds locally
        # with a sentence-transformers model (runs on Metal on Apple Silicon;
        # ~130 MB download on first use, no key required).
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.hf_embed_model)

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r} "
        "(expected 'anthropic', 'openai', 'groq', or 'ollama')"
    )
