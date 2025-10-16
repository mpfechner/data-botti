import pandas as pd
from services.data_summary import _looks_like_datetime, _parse_datetime_series



def compute_generic_insights(df: pd.DataFrame) -> dict:
    """Generische Auto-Insights für beliebige CSVs: Missing, Zeitspalten, Numerik, Kategorien."""
    insights: dict = {}

    # Fehlwerte
    try:
        missing_per_col = df.isna().sum()
        insights["missing"] = {
            "rows_with_missing": int(df.isna().any(axis=1).sum()),
            "per_column": {str(k): int(v) for k, v in missing_per_col.items()},
        }
    except Exception:
        pass

    # Zeitspalten (Datetime)
    dt_info = {}
    dt_warnings: list[str] = []
    for c in df.columns:
        try:
            if not _looks_like_datetime(df[c]):
                continue
            s = _parse_datetime_series(df[c])
            if s.isna().mean() > 0.5:
                # überwiegend NaT → nicht als Datum behandeln
                dt_warnings.append(
                    f"Spalte '{c}': uneinheitliches/unklares Datumsformat – wurde als Text behandelt. "
                    "Tipp: Export in ein konsistentes Format (z. B. ISO 8601 'YYYY-MM-DD') und ohne Mischformen."
                )
                continue
            dt_info[c] = {"min": str(s.min()), "max": str(s.max())}
        except Exception:
            continue
    if dt_info:
        insights["datetime"] = dt_info
    if dt_warnings:
        insights.setdefault("warnings", []).extend(dt_warnings)

    # Numerische Spalten
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    num_info = {}
    for c in num_cols:
        try:
            s = pd.to_numeric(df[c], errors="coerce")
            if not s.notna().any():
                continue
            q = s.quantile([0.25, 0.5, 0.75])
            num_info[c] = {
                "count": int(s.count()),
                "mean": float(s.mean()),
                "std": float(s.std()) if s.count() > 1 else None,
                "min": float(s.min()),
                "q25": float(q.loc[0.25]),
                "median": float(q.loc[0.5]),
                "q75": float(q.loc[0.75]),
                "max": float(s.max()),
            }
        except Exception:
            continue
    if num_info:
        insights["numeric"] = num_info

    # Kategorische Spalten (alles, was nicht numeric/datetime ist)
    cat_info = {}
    cat_cols = [c for c in df.columns if c not in num_cols and c not in (insights.get("datetime") or {}).keys()]
    for c in cat_cols:
        try:
            s = df[c].astype("string")
            if not s.notna().any():
                continue
            vc = s.value_counts(dropna=True)
            top = str(vc.index[0]) if len(vc) > 0 else None
            freq = int(vc.iloc[0]) if len(vc) > 0 else None
            cat_info[c] = {
                "count": int(s.count()),
                "unique": int(s.nunique(dropna=True)),
                "top": top,
                "freq": freq,
            }
        except Exception:
            continue
    if cat_info:
        insights["categorical"] = cat_info

    return insights