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

# Local router deciding the cheapest suitable model
from .ai_router import choose_model  # local router deciding the cheapest suitable model


class AIClientNotConfigured(RuntimeError):
    """Raised when the AI client is not yet wired/configured (e.g., missing API key)."""


# Helper to estimate cache ratio (placeholder)
def _estimate_cache_ratio(context_id: Optional[str]) -> float:
    """Very small placeholder: returns 0.0 if we don't know.
    Later you can wire this to your session/context tracking.
    """
    return 0.0 if not context_id else 0.9  # assume high reuse if an id is present


def ask_model(
    prompt: str,
    *,
    expected_output: str = "medium",
    context_id: Optional[str] = None,
    **kwargs: Any,
) -> str:
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

    # --- Routing: decide model based on expected output and (estimated) cache ratio
    cache_ratio = _estimate_cache_ratio(context_id)
    routed_model = choose_model(expected_output, cache_ratio)

    # Allow manual override via kwargs["model"], otherwise use routed_model
    model = kwargs.get("model", routed_model)

    # Map expected_output to max tokens unless explicitly overridden
    default_max = {"short": 600, "medium": 800, "long": 1200}.get(expected_output, 800)
    # ensure a sensible floor so the model returns something
    max_tokens = max(64, int(kwargs.get("max_tokens", default_max)))
    # Temperature handling: default 1.0 (broadly supported); only send if not 1.0
    _temp_kw = kwargs.get("temperature", None)
    if _temp_kw is None:
        temperature = 1.0
    else:
        temperature = float(_temp_kw)
    system_prompt = kwargs.get(
        "system_prompt",
        "You are a helpful data assistant for CSV/report analysis. Answer clearly and concisely.",
    )

    # --- OpenAI call (lazy import keeps optional dependency) -----------------
    try:
        # Import only when needed so the rest of the app works without the package
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=openai_key)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        params = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if temperature != 1.0:
            params["temperature"] = temperature
        params["response_format"] = {"type": "text"}


        resp = client.chat.completions.create(**params)

        #print("DEBUG raw response:", resp)

        content = ""
        try:
            choice0 = resp.choices[0]
            # Newer SDKs: message.content may be str or list of content parts
            msg = getattr(choice0, "message", None)
            if msg is not None:
                mc = getattr(msg, "content", "")
                if isinstance(mc, str):
                    content = mc.strip()
                elif isinstance(mc, list):
                    parts = []
                    for p in mc:
                        # handle dicts {"type":"text","text":"..."} or objects with .text
                        if isinstance(p, dict):
                            t = p.get("text") or ""
                            if t:
                                parts.append(str(t))
                        else:
                            t = getattr(p, "text", None)
                            if t:
                                parts.append(str(t))
                    content = "\n".join(parts).strip()
            # Legacy fallback: some clients expose .text directly
            if not content and hasattr(choice0, "text"):
                content = (choice0.text or "").strip()
        except Exception:
            content = ""

        if not content:
            # One retry with larger budget and a safer model for short tasks
            try:
                fallback_model = model
                if expected_output == "short" and not str(model).startswith("gpt-4o-mini"):
                    fallback_model = "gpt-4o-mini"
                params2 = {
                    "model": fallback_model,
                    "messages": messages,
                    "max_completion_tokens": max(max_tokens, 600),
                    "response_format": {"type": "text"},
                }
                if temperature != 1.0:
                    params2["temperature"] = temperature

                resp2 = client.chat.completions.create(**params2)

                #print("DEBUG raw response:", resp2)

                content2 = ""
                try:
                    choice0b = resp2.choices[0]
                    msg2 = getattr(choice0b, "message", None)
                    if msg2 is not None:
                        mc2 = getattr(msg2, "content", "")
                        if isinstance(mc2, str):
                            content2 = mc2.strip()
                        elif isinstance(mc2, list):
                            parts2 = []
                            for p in mc2:
                                if isinstance(p, dict):
                                    t = p.get("text") or ""
                                    if t:
                                        parts2.append(str(t))
                                else:
                                    t = getattr(p, "text", None)
                                    if t:
                                        parts2.append(str(t))
                            content2 = "\n".join(parts2).strip()
                    if not content2 and hasattr(choice0b, "text"):
                        content2 = (choice0b.text or "").strip()
                except Exception:
                    content2 = ""

                if content2:
                    return content2
            except Exception:
                pass

            return "(AI returned no textual content. Bitte versuche es erneut oder erhöhe die erwartete Ausgabelänge.)"
        return content

    except ModuleNotFoundError as e:
        # openai package not installed
        return (
            f"(AI error) OpenAI client library not installed: {e}. "
            f"Model chosen: {model}. Please `pip install openai` to enable real calls."
        )
    except Exception as e:
        # Bubble up with a clear message; the caller can render it nicely.
        raise e