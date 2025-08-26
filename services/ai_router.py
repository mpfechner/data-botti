def choose_model(expected_output: str, cache_ratio: float) -> str:
    if expected_output in ("short", "medium") and cache_ratio >= 0.8:
        return "gpt-5-mini"
    if expected_output == "long":
        return "gpt-4o-mini"
    return "gpt-4o-mini"  # Default fallback