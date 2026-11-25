"""Model resolution helper supporting live LLM strings and offline FunctionModel."""

from __future__ import annotations

import os
from typing import Callable

from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.messages import ModelMessage, ModelResponse


_PROVIDER_KEY_MAP: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "google-gla": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "google-vertex": ("GOOGLE_APPLICATION_CREDENTIALS", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
}


def resolve_model_choice(requested_model: str | None = None) -> tuple[str, bool]:
    """Resolve requested model string and whether offline deterministic fallback applies.

    Returns ``(effective_model_label, use_deterministic)``.
    """
    raw = (requested_model or os.getenv("PROCUREMENT_MODEL") or "deterministic").strip()
    if not raw or raw.lower() in {"deterministic", "offline", "function"}:
        return "deterministic", True

    provider = raw.split(":", 1)[0].strip().lower() if ":" in raw else ""
    required_keys = _PROVIDER_KEY_MAP.get(provider)
    if required_keys and not any(os.getenv(k, "").strip() for k in required_keys):
        return f"deterministic (fallback from {raw}: missing API key)", True

    return raw, False


def build_agent_model(
    requested_model: str | None,
    deterministic_fn: Callable[[list[ModelMessage], AgentInfo], ModelResponse],
) -> tuple[str | FunctionModel, str]:
    """Return either a PydanticAI FunctionModel or a live model string, plus display name."""
    label, use_deterministic = resolve_model_choice(requested_model)
    if use_deterministic:
        return FunctionModel(deterministic_fn), label
    return label, label
