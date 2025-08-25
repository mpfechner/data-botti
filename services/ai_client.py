"""
services.ai_client
Low-level AI client wrapper.

Responsibility:
- Hold provider-specific call logic (e.g., OpenAI)
- Expose a stable function: ask_model(prompt: str, **kwargs) -> str
- Keep secrets in environment variables, never hardcode
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any


class AIClientNotConfigured(RuntimeError):
    """Raised when the AI client is not yet wired/configured (e.g., missing API key)."""


def ask_model(prompt: str, **kwargs: Any) -> str:
    """
    Minimal façade for calling an LLM.

    Parameters
    ----------
    prompt : str
        The full prompt to send to the model/provider.
    **kwargs : Any
        Optional provider-specific overrides such as:
        - model: str
        - temperature: float
        - max_tokens: int
        - system_prompt: str
        - etc.

    Returns
    -------
    str
        The model's textual answer.

    Raises
    ------
    AIClientNotConfigured
        If no provider is configured (e.g., missing API key).
    Exception
        For provider-specific errors.
    """
    # Example: prefer OPENAI_* env vars; if none present, raise a clear error.
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_BOTTI")

    if not openai_key:
        raise AIClientNotConfigured(
            "No AI provider configured. Set OPENAI_API_KEY (or OPENAI_API_KEY_BOTTI) in your environment."
        )

    # --- OpenAI example (lazy import to avoid hard dependency if unused) -------
    try:
        # If you use the official OpenAI python client:
        # from openai import OpenAI
        # client = OpenAI(api_key=openai_key)
        #
        # model = kwargs.get("model", "gpt-4o-mini")  # pick your default
        # system_prompt = kwargs.get("system_prompt", "You are a helpful data assistant for CSV analysis.")
        # temperature = float(kwargs.get("temperature", 0.2))
        # max_tokens = int(kwargs.get("max_tokens", 800))
        #
        # resp = client.chat.completions.create(
        #     model=model,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": prompt},
        #     ],
        #     temperature=temperature,
        #     max_tokens=max_tokens,
        # )
        # return resp.choices[0].message.content.strip()

        # Placeholder until you wire the real client:
        return "(AI placeholder) The AI client is configured (API key present), but the provider call is not yet implemented."
    except Exception as e:
        # Bubble up with a clear message; the caller can render it nicely.
        raise