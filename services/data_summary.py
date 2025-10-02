from __future__ import annotations

# Temporary shim for datetime helpers.
# Delegates to the legacy implementations in helpers.py so we can
# move import sites now and inline later without behavior changes.
try:
    from helpers import (
        _looks_like_datetime as _legacy_looks_like_datetime,  # type: ignore[attr-defined]
        _parse_datetime_series as _legacy_parse_datetime_series,  # type: ignore[attr-defined]
    )
    _import_error: Exception | None = None
except Exception as _e:  # pragma: no cover
    _legacy_looks_like_datetime = None
    _legacy_parse_datetime_series = None
    _import_error = _e


def _looks_like_datetime(series) -> bool:
    if _legacy_looks_like_datetime is None:  # pragma: no cover
        raise ImportError(
            f"Could not import legacy _looks_like_datetime from helpers.py: {_import_error}"
        )
    return _legacy_looks_like_datetime(series)


def _parse_datetime_series(series):
    if _legacy_parse_datetime_series is None:  # pragma: no cover
        raise ImportError(
            f"Could not import legacy _parse_datetime_series from helpers.py: {_import_error}"
        )
    return _legacy_parse_datetime_series(series)
