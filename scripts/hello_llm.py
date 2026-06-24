"""Phase 0 smoke test: stream a reply from the configured LLM provider.

This is the "hello, world" of the project. It proves three things at once:
  1. Your environment / API key is set up correctly.
  2. The provider abstraction works (Claude vs OpenAI is just a config switch).
  3. You can stream tokens from the model as they're generated.

Run it from the REPO ROOT (the folder containing `src/` and `scripts/`):

    python -m scripts.hello_llm
    python -m scripts.hello_llm "Explain what a 10-K filing is in two sentences."

Switch providers by editing LLM_PROVIDER in your .env (anthropic | openai),
then run it again — the code does not change.
"""

from __future__ import annotations

import sys

from src.config import get_settings
from src.llm.factory import get_chat_model


def _check_api_key(settings) -> str | None:
    """Return a friendly error string if the selected provider's key is missing."""
    if settings.llm_provider == "anthropic" and not settings.anthropic_api_key:
        return (
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your "
            "Anthropic key (or set LLM_PROVIDER=openai)."
        )
    if settings.llm_provider == "openai" and not settings.openai_api_key:
        return (
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your "
            "OpenAI key (or set LLM_PROVIDER=anthropic)."
        )
    return None


def main() -> int:
    settings = get_settings()

    error = _check_api_key(settings)
    if error:
        print(f"[config error] {error}", file=sys.stderr)
        return 1

    prompt = " ".join(sys.argv[1:]).strip() or (
        "In one sentence, what is a financial 10-K filing?"
    )

    print(f"Provider : {settings.llm_provider}")
    print(f"Model    : {settings.chat_model_name}")
    print(f"Prompt   : {prompt}")
    print("-" * 60)

    chat = get_chat_model(settings)

    # .stream() yields message chunks as the model generates them. Printing each
    # chunk's text with flush=True shows tokens appearing live in your terminal.
    for chunk in chat.stream(prompt):
        print(chunk.content, end="", flush=True)
    print()  # final newline

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
