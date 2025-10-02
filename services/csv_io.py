

from __future__ import annotations
from typing import Optional

# Temporary shim: route calls via the legacy implementation in helpers.py.
# This lets us change import sites to services.csv_io without touching logic yet.
try:
    from helpers import load_csv_resilient as _legacy_load_csv_resilient  # type: ignore[attr-defined]
    _import_error: Exception | None = None
except Exception as _e:  # pragma: no cover
    _legacy_load_csv_resilient = None
    _import_error = _e


def load_csv_resilient(
    file_path: str,
    preferred_encoding: Optional[str] = None,
    preferred_delimiter: Optional[str] = None,
):
    """
    Thin wrapper delegating to the legacy helpers.load_csv_resilient.
    Returns (df, used_delimiter, used_encoding) exactly as before.

    Phase 1 step: keep behavior identical while moving import sites to services.csv_io.
    Later we inline the original implementation here and remove the helpers import.
    """
    if _legacy_load_csv_resilient is None:  # pragma: no cover
        raise ImportError(
            f"Could not import legacy load_csv_resilient from helpers.py: {_import_error}"
        )
    return _legacy_load_csv_resilient(
        file_path,
        preferred_encoding,
        preferred_delimiter,
    )