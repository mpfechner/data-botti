from __future__ import annotations

from typing import Dict

import logging

try:
    from infra.config import get_config
    _cfg = get_config()
    MODEL_FAST = _cfg.get("AI_MODEL_FAST") or "gpt-4o-mini"
    MODEL_SMART = _cfg.get("AI_MODEL_SMART") or "gpt-5-mini"
    FORCE_MODEL = _cfg.get("AI_FORCE_MODEL") or ""
    try:
        DOWNGRADE_SAVING = float(_cfg.get("AI_DOWNGRADE_SAVING", 0.6))
    except Exception:
        DOWNGRADE_SAVING = 0.6
except Exception:
    # Safe fallbacks if config is not wired (e.g., during import-time in tests)
    MODEL_FAST = "gpt-4o-mini"
    MODEL_SMART = "gpt-5-mini"
    FORCE_MODEL = ""
    DOWNGRADE_SAVING = 0.6

# Price table (USD per 1K tokens). Values based on the table Michael provided.
# Note: Pricing for gpt-5o was not explicitly listed; we treat it like the gpt-5 tier for now.
PRICES: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini":   {"in": 0.15,  "cached": 0.075, "out": 0.60},
    "gpt-4.1-mini":  {"in": 0.40,  "cached": 0.10,  "out": 1.60},
    "gpt-5-mini":        {"in": 1.25,  "cached": 0.125, "out": 10.00},  # adjust if your env differs
}

# Quality preference order (newest first)
PREF_ORDER = ["gpt-5-mini", "gpt-4.1-mini", "gpt-4o-mini"]

# Default token estimates if caller does not provide concrete numbers
DEFAULT_OUT_TOKENS = {"short": 300, "medium": 800, "long": 2000}
DEFAULT_IN_TOKENS = {"short": 1500, "medium": 3000, "long": 6000}


def _estimate_tokens(expected_output: str, est_in_tokens: int | None, est_out_tokens: int | None) -> tuple[int, int]:
    eo = (expected_output or "medium").lower()
    eo = eo if eo in DEFAULT_OUT_TOKENS else "medium"
    out_tokens = est_out_tokens if est_out_tokens is not None else DEFAULT_OUT_TOKENS[eo]
    in_tokens = est_in_tokens if est_in_tokens is not None else DEFAULT_IN_TOKENS[eo]
    return in_tokens, out_tokens


def _total_cost(model: str, in_tokens: int, out_tokens: int, cache_ratio: float) -> float:
    p = PRICES[model]
    # blended input price based on cache hit ratio
    blended_in = (1.0 - cache_ratio) * p["in"] + cache_ratio * p["cached"]
    # Prices are per 1K tokens
    return (in_tokens / 1000.0) * blended_in + (out_tokens / 1000.0) * p["out"]


def choose_model(
    expected_output: str | None,
    cache_ratio: float | None,
    *,
    prompt: str | None = None,
    context: str = "",
    est_in_tokens: int | None = None,
    est_out_tokens: int | None = None,
    price_delta_pref: float = 0.15,
) -> str:
    if expected_output is None and prompt is not None:
        expected_output = infer_expected_output(prompt, context=context)
    """
    Choose model with a quality‑first strategy; downgrade only if savings exceed a threshold.

    Args:
        expected_output: one of {"short","medium","long"}; others normalize to "medium".
        cache_ratio: 0.0..1.0; fraction of input tokens served from cache. None -> assume 0.0.
        est_in_tokens: optional explicit estimate of input tokens; default based on expected_output.
        est_out_tokens: optional explicit estimate of output tokens; default based on expected_output.
        price_delta_pref: if the cheapest and a newer model are within this fractional delta,
                          prefer the newer model (quality tilt).

    Returns:
        The chosen model name as string.
    """
    if FORCE_MODEL:
        return FORCE_MODEL

    cache = max(0.0, min(1.0, cache_ratio or 0.0))
    in_tokens, out_tokens = _estimate_tokens(expected_output, est_in_tokens, est_out_tokens)

    candidates = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini"]
    costs = {m: _total_cost(m, in_tokens, out_tokens, cache) for m in candidates}

    # Quality-first: start with the newest/best preferred model
    preferred = next((m for m in PREF_ORDER if m in costs), "gpt-4o-mini")
    preferred_cost = costs[preferred]

    # Cheapest model and its cost
    cheapest = min(candidates, key=lambda m: costs[m])
    cheapest_cost = costs[cheapest]

    logger = logging.getLogger("ai_router")
    logger.info(
        "Routing decision: expected_output=%s cache_ratio=%.2f in_tokens=%s out_tokens=%s preferred=%s cost_preferred=%.4f cheapest=%s cost_cheapest=%.4f saving=%.2f%% threshold=%.2f",
        expected_output,
        cache,
        in_tokens,
        out_tokens,
        preferred,
        preferred_cost,
        cheapest,
        cheapest_cost,
        (1.0 - (cheapest_cost / max(preferred_cost, 1e-9))) * 100 if cheapest_cost < preferred_cost else 0.0,
        DOWNGRADE_SAVING * 100,
    )

    # If the cheapest option yields a sufficiently large saving vs preferred, downgrade
    # Saving is defined as 1 - (cheapest / preferred). Example: preferred=$10, cheapest=$4 -> saving=60%.
    if cheapest_cost < preferred_cost:
        saving = 1.0 - (cheapest_cost / max(preferred_cost, 1e-9))
        if saving >= DOWNGRADE_SAVING:
            return cheapest

    return preferred


import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def infer_expected_output(prompt: str, context: str = "", model: str = "gpt-4") -> str:
    token_count = count_tokens(prompt + " " + context, model=model)
    if token_count < 100:
        return "short"
    elif token_count < 300:
        return "medium"
    return "long"
