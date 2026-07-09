"""
Thin wrapper around the Qwen LLM API.

Qwen (Alibaba Cloud / DashScope) exposes an OpenAI-compatible endpoint,
so we reuse the official `openai` Python SDK pointed at Qwen's base
URL. This keeps the rest of the codebase provider-agnostic: swapping
Qwen for any other OpenAI-compatible provider only requires changing
QWEN_BASE_URL / QWEN_MODEL in the .env file.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Optional

from openai import OpenAI

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Raised when the LLM call fails or returns unparsable content."""


@lru_cache
def get_llm_client() -> OpenAI:
    """Return a cached OpenAI-compatible client configured for Qwen."""
    settings = get_settings()
    return OpenAI(api_key=settings.QWEN_API_KEY, base_url=settings.QWEN_BASE_URL)


def call_llm(prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.4) -> str:
    """
    Send a single prompt to the Qwen model and return the raw text response.

    Args:
        prompt: The user prompt (already fully built by a prompts/*.py function).
        system_prompt: Optional system-level instruction.
        temperature: Sampling temperature. Lower = more deterministic
            (important for JSON-formatted outputs).

    Raises:
        LLMClientError: if the API call fails for any reason.
    """
    settings = get_settings()
    client = get_llm_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=settings.QWEN_MODEL,
            messages=messages,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            raise LLMClientError("LLM returned an empty response.")
        return content.strip()
    except Exception as exc:  # noqa: BLE001 - we deliberately wrap every failure
        logger.exception("LLM call failed")
        raise LLMClientError(f"Failed to call Qwen LLM: {exc}") from exc


def safe_json_parse(raw_text: str) -> dict[str, Any]:
    """
    Parse a JSON object out of the raw LLM response, tolerating common
    formatting mistakes (e.g. accidental markdown code fences).

    Raises:
        LLMClientError: if the text cannot be parsed as JSON.
    """
    cleaned = raw_text.strip()

    # Strip accidental markdown code fences if the model added them anyway
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse LLM JSON output: %s", raw_text)
        raise LLMClientError(f"LLM did not return valid JSON: {exc}") from exc
