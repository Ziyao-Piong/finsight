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
    llm_provider: Literal["anthropic", "openai"] = "anthropic"

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

    # --- Generation knobs ---
    # Temperature is intentionally optional. Newer Claude models (Opus 4.7/4.8,
    # Fable 5) reject a temperature parameter, so we only pass it when a value is
    # explicitly set — keeping provider/model swaps safe by default.
    llm_temperature: float | None = None
    llm_max_tokens: int = 1024

    # --- Storage (used from Phase 1 onward) ---
    chroma_dir: str = ".chroma"

    @property
    def chat_model_name(self) -> str:
        """The chat model id for the currently selected provider."""
        return (
            self.anthropic_chat_model
            if self.llm_provider == "anthropic"
            else self.openai_chat_model
        )

    @property
    def embed_model_name(self) -> str:
        """The embeddings model id for the currently selected provider."""
        return (
            self.voyage_embed_model
            if self.llm_provider == "anthropic"
            else self.openai_embed_model
        )


def _mirror_keys_to_env(settings: Settings) -> None:
    """Copy provider API keys into ``os.environ``.

    LangChain integrations look up their keys from standard environment
    variables (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``VOYAGE_API_KEY``).
    When a key comes from the ``.env`` file, ``pydantic-settings`` loads it into
    the Settings object but NOT into ``os.environ`` — so we mirror it here. This
    way the same code works whether you exported the key in your shell or put it
    in ``.env``. We never overwrite a value already present in the environment.
    """
    for env_name, value in {
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
        "OPENAI_API_KEY": settings.openai_api_key,
        "VOYAGE_API_KEY": settings.voyage_api_key,
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
