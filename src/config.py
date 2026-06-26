"""Central configuration for FinSight.

All tunable values (which LLM provider to use, model ids, API keys, paths) live
here and are loaded from environment variables and/or a local ``.env`` file via
``pydantic-settings``. Nothing else in the codebase reads ``os.environ`` directly
for these values — they ask :func:`get_settings` instead. That keeps configuration
in one obvious place and makes the rest of the code testable.

Concepts demonstrated here:
  * 12-factor style config — secrets come from the environment, never hard-coded.
  * ``pydantic-settings`` — typed settings with validation and ``.env`` support.
  * A single source of truth for the provider switch (``LLM_PROVIDER``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, populated from env vars / ``.env``.

    Field names map to UPPER_CASE env var names automatically and
    case-insensitively (e.g. ``llm_provider`` <- ``LLM_PROVIDER``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated env vars instead of erroring
    )

    # Which provider powers BOTH the chat model and the embeddings.
    #   anthropic / openai -> frontier (hosted, paid).
    #   groq               -> hosted open-source (free tier, OpenAI-compatible API).
    #   ollama             -> offline open-source running locally on your machine.
    # Having all four behind one switch is what lets the Phase 5 eval harness
    # compare frontier vs. hosted-OSS vs. local-OSS on the same questions.
    llm_provider: Literal["anthropic", "openai", "groq", "ollama"] = "anthropic"

    # --- Anthropic (Claude) ---
    anthropic_api_key: str | None = None
    anthropic_chat_model: str = "claude-sonnet-4-6"

    # --- OpenAI ---
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    # --- Voyage AI (embeddings for the Anthropic path) ---
    voyage_api_key: str | None = None
    voyage_embed_model: str = "voyage-3"

    # --- Groq (hosted open-source chat) ---
    # Groq serves open models (Llama, etc.) behind an OpenAI-COMPATIBLE API, so the
    # factory reuses ChatOpenAI and just points it at groq_base_url. Keeping the
    # branch generic means switching to OpenRouter/Together later is only an env
    # change (different base_url + key + model id), not new code. Groq has no
    # embeddings API, so the Groq path pairs with local HuggingFace embeddings below.
    groq_api_key: str | None = None
    # Strong tool-use OSS model on Groq's free tier. NOTE: Groq rotates model ids —
    # check https://console.groq.com/docs/models if this 400s.
    groq_chat_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # --- Ollama (offline open-source, runs locally) ---
    # No API key needed — talks to a local Ollama server. qwen2.5:7b has strong
    # tool-use for its size and fits an M2 Pro comfortably; llama3.1:8b is an
    # alternative, and qwen2.5:14b is viable on 32 GB machines. Pull models first:
    #   ollama pull qwen2.5:7b && ollama pull nomic-embed-text
    ollama_chat_model: str = "qwen2.5:7b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"

    # --- HuggingFace embeddings (local, used by the Groq path) ---
    # Sentence-transformers model that runs on-device (Metal on Apple Silicon),
    # free and no key required. ~130 MB download on first use.
    hf_embed_model: str = "BAAI/bge-small-en-v1.5"

    # --- Generation knobs ---
    # Temperature is intentionally optional. Newer Claude models (Opus 4.7/4.8,
    # Fable 5) reject a temperature parameter, so we only pass it when a value is
    # explicitly set — keeping provider/model swaps safe by default.
    llm_temperature: float | None = None
    llm_max_tokens: int = 1024

    # --- Storage (used from Phase 1 onward) ---
    chroma_dir: str = ".chroma"

    # --- Retrieval / chunking (Phase 1) ---
    # Naive fixed-size chunking knobs. chunk_size is in characters (not tokens) —
    # RecursiveCharacterTextSplitter counts characters by default. chunk_overlap
    # repeats a slice of the previous chunk so an answer that straddles a boundary
    # isn't split in half. retrieval_k is how many chunks we stuff into the prompt.
    # These are deliberately simple; Phase 2/3 revisit chunking and retrieval.
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_k: int = 4

    # SEC EDGAR requires a descriptive User-Agent that identifies you with contact
    # info; requests without one are throttled or rejected (403). See
    # https://www.sec.gov/os/webmaster-faq#developers .
    edgar_user_agent: str = "FinSight learning project ziyaopiong@gmail.com"

    @property
    def collection_name(self) -> str:
        """Chroma collection name, keyed by provider + embedding model.

        The embedding model defines the vector space, so two providers' vectors
        must never share a collection (their numbers aren't comparable). Keying the
        name on the provider and embed model gives each its own collection and makes
        the "switching providers requires re-ingesting" rule structural rather than
        a thing you have to remember.
        """
        slug = self.embed_model_name.replace("/", "_").replace(":", "_")
        return f"finsight_{self.llm_provider}_{slug}"

    @property
    def chat_model_name(self) -> str:
        """The chat model id for the currently selected provider."""
        return {
            "anthropic": self.anthropic_chat_model,
            "openai": self.openai_chat_model,
            "groq": self.groq_chat_model,
            "ollama": self.ollama_chat_model,
        }[self.llm_provider]

    @property
    def embed_model_name(self) -> str:
        """The embeddings model id for the currently selected provider.

        This feeds the per-provider Chroma collection name (Phase 1+). Each
        provider has its OWN embedding model and therefore its own vector space,
        so keying the collection on this value keeps the four spaces from mixing.
        """
        return {
            "anthropic": self.voyage_embed_model,
            "openai": self.openai_embed_model,
            "groq": self.hf_embed_model,  # Groq has no embeddings -> local HF
            "ollama": self.ollama_embed_model,
        }[self.llm_provider]


def _mirror_keys_to_env(settings: Settings) -> None:
    """Copy provider API keys into ``os.environ``.

    LangChain integrations look up their keys from standard environment
    variables (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``VOYAGE_API_KEY``).
    When a key comes from the ``.env`` file, ``pydantic-settings`` loads it into
    the Settings object but NOT into ``os.environ`` — so we mirror it here. This
    way the same code works whether you exported the key in your shell or put it
    in ``.env``. We never overwrite a value already present in the environment.
    """
    # Ollama needs no key (local server); public HuggingFace models need none either.
    for env_name, value in {
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
        "OPENAI_API_KEY": settings.openai_api_key,
        "VOYAGE_API_KEY": settings.voyage_api_key,
        "GROQ_API_KEY": settings.groq_api_key,
    }.items():
        if value and not os.environ.get(env_name):
            os.environ[env_name] = value


@lru_cache
def get_settings() -> Settings:
    """Return the (cached) application settings.

    Cached so the ``.env`` file is read once per process and every caller shares
    the same configuration object.
    """
    settings = Settings()
    _mirror_keys_to_env(settings)
    return settings
