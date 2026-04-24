"""Minimal LLM client helpers for provider detection and configuration."""

from __future__ import annotations

import os
from typing import Any


def infer_provider() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("AZURE_OPENAI_API_KEY"):
        return "azure"
    return "none"


def default_model_for_provider(provider: str) -> str:
    defaults = {
        "openai": "gpt-4o",
        "azure": "gpt-4o",
    }
    return defaults.get(provider, "gpt-4o")


def resolve_client_kwargs(*, provider: str = "auto") -> dict[str, Any]:
    if provider == "auto":
        provider = infer_provider()
    kwargs: dict[str, Any] = {}
    if provider == "azure":
        kwargs["api_key"] = os.environ.get("AZURE_OPENAI_API_KEY", "")
        kwargs["base_url"] = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            kwargs["api_key"] = api_key
        base_url = os.environ.get("OPENAI_BASE_URL", "")
        if base_url:
            kwargs["base_url"] = base_url
    return kwargs
