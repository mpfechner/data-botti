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
import time
from typing import Optional, Dict, Any
from infra.config import get_config

import logging
logger = logging.getLogger(__name__)

from services.models import TokenUsage

# Local router deciding the cheapest suitable model
from .ai_router import choose_model  # local router deciding the cheapest suitable model


class AIClientNotConfigured(RuntimeError):
    """Raised when the AI client is not yet wired/configured (e.g., missing API key)."""


def is_ai_available() -> bool:
    """Return True if an AI provider is configured (API key present)."""
    config = get_config()
    return bool(config["OPENAI_API_KEY"] or config["OPENAI_API_KEY_BOTTI"])


# Helper to estimate cache ratio (placeholder)
def _estimate_cache_ratio(context_id: Optional[str]) -> float:
    """Very small placeholder: returns 0.0 if we don't know.
    Later you can wire this to your session/context tracking.
    """
    return 0.0 if not context_id else 0.9  # assume high reuse if an id is present


def call_model(*, model: str, messages: list[dict], max_tokens: int = 800, temperature: float | None = None) -> tuple[str, TokenUsage]:
    """Low-level call: send chat messages to a specific model and return textual content."""
    config = get_config()
    openai_key = config["OPENAI_API_KEY"] or config["OPENAI_API_KEY_BOTTI"]
    if not openai_key:
        raise AIClientNotConfigured(
            "No AI provider configured. Set OPENAI_API_KEY (or OPENAI_API_KEY_BOTTI) in your environment."
        )

    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=openai_key)

        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        # Newer GPT‑5 family expects `max_completion_tokens` instead of `max_tokens`.
        if str(model).startswith("gpt-5"):
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

        # Temperature handling: GPT‑5 family does not support custom temperatures (only default 1). Omit param.
        if str(model).startswith("gpt-5"):
            pass  # do not include temperature for GPT‑5 models
        else:
            if temperature is not None and temperature != 1.0:
                params["temperature"] = float(temperature)

        # --- perf: measure model roundtrip time (TTLB) ---
        _perf_enabled = os.environ.get("AI_TIMING", "1") != "0"
        _t_start = time.monotonic() if _perf_enabled else None
        resp = client.chat.completions.create(**params)
        if _perf_enabled and _t_start is not None:
            _t_end = time.monotonic()
            _ms = (_t_end - _t_start) * 1000.0
            try:
                _rid = getattr(resp, "id", None) or getattr(resp, "response", None)
            except Exception:
                _rid = None
            logger.info(
                "[perf] openai_ttlb_ms=%.1f model=%s max_tokens=%s temperature=%s rid=%s",
                _ms,
                model,
                params.get("max_completion_tokens") or params.get("max_tokens"),
                None if str(model).startswith("gpt-5") else (None if (temperature is None or float(temperature) == 1.0) else float(temperature)),
                _rid,
            )

        usage = TokenUsage()
        try:
            if hasattr(resp, "usage"):
                u = resp.usage
                usage.prompt_tokens = getattr(u, "prompt_tokens", 0) or 0
                usage.completion_tokens = getattr(u, "completion_tokens", 0) or 0
                usage.total_tokens = getattr(u, "total_tokens", 0) or usage.prompt_tokens + usage.completion_tokens
        except Exception:
            pass

        # --- Debug: write raw response to file ---
        try:
            import json, datetime, pathlib
            debug_dir = pathlib.Path("logs/debug_ai")
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = debug_dir / f"resp_{model}_{ts}.json"
            if hasattr(resp, "to_dict_recursive"):
                data = resp.to_dict_recursive()
            elif hasattr(resp, "to_dict"):
                data = resp.to_dict()
            else:
                try:
                    data = dict(resp)
                except Exception:
                    data = {"repr": repr(resp)}
            with open(fname, "w", encoding="utf-8") as f:
                f.write(json.dumps(data, indent=2, ensure_ascii=False))
            logger.info("Wrote raw AI response to %s", fname)
        except Exception as e:
            logger.warning("Failed to write raw AI response: %s", e)

        content = ""
        try:
            choice0 = resp.choices[0]
            msg = getattr(choice0, "message", None)
            if msg is not None:
                mc = getattr(msg, "content", "")
                if isinstance(mc, str):
                    content = mc.strip()
                elif isinstance(mc, list):
                    parts = []
                    for p in mc:
                        if isinstance(p, dict):
                            t = p.get("text") or ""
                            if t:
                                parts.append(str(t))
                        else:
                            t = getattr(p, "text", None)
                            if t:
                                parts.append(str(t))
                    content = "\n".join(parts).strip()
            if not content and hasattr(choice0, "text"):
                content = (choice0.text or "").strip()
        except Exception:
            content = ""

        if not content:
            # GPT‑5 family sometimes returns plain text in `output_text` or similar fields
            if hasattr(resp, "output_text"):
                content = (getattr(resp, "output_text") or "").strip()
            elif isinstance(resp, dict) and "output_text" in resp:
                content = (resp.get("output_text") or "").strip()

        if not content:
            return "(AI returned no textual content. Bitte versuche es erneut oder erhöhe die erwartete Ausgabelänge.)", usage
        return content, usage

    except ModuleNotFoundError as e:
        usage = TokenUsage()
        return (
            f"(AI error) OpenAI client library not installed: {e}. "
            f"Model chosen: {model}. Please `pip install openai` to enable real calls."
        ), usage
    except Exception as e:
        raise e


def ask_model(
    prompt: str,
    *,
    expected_output: str = "medium",
    context_id: Optional[str] = None,
    **kwargs: Any,
) -> tuple[str, TokenUsage]:
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
    config = get_config()
    openai_key = config["OPENAI_API_KEY"] or config["OPENAI_API_KEY_BOTTI"]

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

    # Build messages and delegate to call_model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    _perf_enabled = os.environ.get("AI_TIMING", "1") != "0"
    _t_overall_start = time.monotonic() if _perf_enabled else None
    content, usage = call_model(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    if _perf_enabled and _t_overall_start is not None:
        _t_overall_end = time.monotonic()
        logger.info(
            "[perf] ask_model_total_ms=%.1f routed_model=%s expected_output=%s prompt_chars=%d",
            (_t_overall_end - _t_overall_start) * 1000.0,
            model,
            expected_output,
            len(prompt or ""),
        )

    if content.startswith("(AI error)") or content.startswith("(AI returned no textual content"):
        # One retry with larger budget and a safer fast model for short tasks
        try:
            fallback_model = model
            if expected_output == "short" and not str(model).startswith("gpt-4o-mini"):
                fallback_model = "gpt-4o-mini"
            content2, usage2 = call_model(
                model=fallback_model,
                messages=messages,
                max_tokens=max(max_tokens, 600),
                temperature=temperature,
            )
            if content2 and not content2.startswith("(AI error)"):
                return content2, usage2
        except Exception:
            pass

    return content, usage
