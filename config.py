"""Shared environment and API key configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"


def load_env() -> None:
    """Load environment variables from the project .env file."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=True)


def get_api_key(explicit: str | None = None) -> str:
    """
    Resolve the Gemini API key from explicit value or environment.

    The project `.env` file takes priority over system environment variables
    so a stale GOOGLE_API_KEY in Windows does not override GEMINI_API_KEY.
    """
    if explicit:
        key = explicit.strip().strip('"').strip("'")
        if not key:
            raise ValueError("Provided API key is empty.")
        return key

    load_env()

    key: str | None = None
    if ENV_PATH.exists():
        file_values = dotenv_values(ENV_PATH)
        key = file_values.get("GEMINI_API_KEY") or file_values.get("GOOGLE_API_KEY")

    if not key:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            f"Add it to {ENV_PATH} or set it in your environment."
        )

    key = key.strip().strip('"').strip("'")
    if not key:
        raise ValueError("GEMINI_API_KEY is empty after trimming whitespace/quotes.")
    return key


def get_api_key_fingerprint() -> str:
    """Return a short fingerprint of the active key for UI diagnostics."""
    key = get_api_key()
    return f"{key[:8]}… ({len(key)} chars)"
