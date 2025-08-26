def choose_model(expected_output: str, cache_ratio: float) -> str:
    eo = (expected_output or "medium").lower()

    # Force short outputs to gpt-4o-mini to avoid reasoning burn on gpt-5-mini
    if eo == "short":
        return "gpt-4o-mini"

    # Medium: prefer 4o-mini
    if eo == "medium":
        return "gpt-4o-mini"

    # Long tasks: try 4o-mini first
    if eo == "long":
        return "gpt-4o-mini"

    # Fallback if nothing else matches
    return "gpt-4.1-mini"