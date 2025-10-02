import gzip
from sqlalchemy import text
import pandas as pd
import re

def _parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parse a Series to datetime robustly:
    1) If already datetime dtype, coerce directly.
    2) Try infer_datetime_format on the full series.
    3) Try a list of common explicit formats and pick the best (>50% parsable).
    4) Fallback to a generic coerce; caller decides on usefulness via NaT ratio.
    """
    s = series
    # 1) Already datetime?
    try:
        if pd.api.types.is_datetime64_any_dtype(s):
            return pd.to_datetime(s, errors="coerce")
    except Exception:
        pass

    # 2) Try explicit formats and choose best
    formats = [
        "%Y-%m-%d",
        "%d.%m.%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%Y.%m.%d",
        "%d.%m.%y",
        "%m/%d/%y",
        "%Y%m%d",
    ]
    best = None
    best_score = 0.0
    for fmt in formats:
        try:
            r = pd.to_datetime(s, format=fmt, errors="coerce")
            score = r.notna().mean()
            if score > best_score:
                best = r
                best_score = score
                if score >= 0.8:  # good enough, stop early
                    break
        except Exception:
            continue
    if best is not None and best_score >= 0.5:
        return best

    # 4) Generic fallback
    # 4) Kein konsistentes Format → nicht als Datum behandeln
    return pd.Series([pd.NaT] * len(s), index=s.index)

def _looks_like_datetime(series: pd.Series) -> bool:
    """
    Schnellheuristik: Nur Spalten prüfen, die wahrscheinlich Datumswerte enthalten.
    - Numerisch: nur dann True, wenn sie wie Unix-Timestamps aussehen (10–13-stellig).
    - Strings: Anteil von "datumstypischen" Tokens (Ziffern + Trenner oder Monatsnamen) >= 0.5.
    """
    s = series.dropna()
    if s.empty:
        return False

    # Numerisch: mögliche Unix-Timestamps (Sekunden/Millis)
    if pd.api.types.is_numeric_dtype(s):
        # 10-stellig (Sekunden) oder 13-stellig (Millis) ohne Dezimalanteil
        as_int = pd.to_numeric(s, errors="coerce").dropna().astype("int64", errors="ignore")
        if as_int.empty:
            return False
        # Heuristik: mind. 50% Werte im plausiblen Bereich
        sec_like = ((as_int >= 946684800) & (as_int <= 4102444800)).mean()  # 2000–2100 (Sekunden)
        ms_like  = ((as_int >= 946684800000) & (as_int <= 4102444800000)).mean()  # 2000–2100 (Millis)
        return max(sec_like, ms_like) >= 0.5

    # Strings: Muster prüfen
    sample = s.astype(str).head(200)
    date_sep_re = re.compile(r"\b\d{1,4}[-./]\d{1,2}[-./]\d{1,4}\b")
    month_names = ("jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec",
                   "januar","februar","märz","maerz","april","mai","juni","juli","august",
                   "september","oktober","november","dezember")
    def looks_token(x: str) -> bool:
        x_low = x.lower()
        return bool(date_sep_re.search(x_low)) or any(m in x_low for m in month_names)

    share = sample.apply(looks_token).mean()
    return share >= 0.5




def analyze_and_store_columns(db_engine, dataset_id: int, df):
    with db_engine.begin() as conn:
        # idempotent: alte Analyse (falls vorhanden) löschen
        conn.execute(text("DELETE FROM dataset_columns WHERE dataset_id = :id"), {"id": dataset_id})

        for idx, col in enumerate(df.columns):
            series = df[col]
            dtype = str(series.dtype)
            is_nullable = 1 if series.isnull().any() else 0
            distinct_count = int(series.nunique(dropna=True))

            min_val = None
            max_val = None
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
                try:
                    min_val = str(series.min(skipna=True))
                    max_val = str(series.max(skipna=True))
                except Exception:
                    pass

            conn.execute(
                text("""
                    INSERT INTO dataset_columns
                      (dataset_id, ordinal, name, dtype, is_nullable, distinct_count, min_val, max_val)
                    VALUES
                      (:dataset_id, :ordinal, :name, :dtype, :is_nullable, :distinct_count, :min_val, :max_val)
                """),
                {
                    "dataset_id": dataset_id,
                    "ordinal": idx,
                    "name": col,
                    "dtype": dtype,
                    "is_nullable": is_nullable,
                    "distinct_count": distinct_count,
                    "min_val": min_val,
                    "max_val": max_val,
                },
            )




def summarize_columns_for_selection(df: pd.DataFrame) -> list[str]:
    """Return ultra-compact per-column summaries for Stage-1 column selection.
    Each line is machine-friendly and type-aware.
    """
    lines: list[str] = []
    n_rows = len(df)

    for col in df.columns:
        s = df[col]
        nonnull = int(s.notna().sum())
        missing_pct = float((1 - (nonnull / max(1, n_rows))) * 100)
        unique = int(s.nunique(dropna=True))

        flags: list[str] = []
        # looks like ID: name heuristic or all values unique
        name_low = str(col).lower()
        if unique == n_rows or any(tok in name_low for tok in ("id", "uuid", "guid")):
            flags.append("looks_id")
        if unique == 1:
            flags.append("constant")
        if missing_pct >= 90.0:
            flags.append("sparse")

        # detect dtype categories
        col_type = "other"
        extra = ""
        try:
            if pd.api.types.is_numeric_dtype(s):
                col_type = "numeric"
                s_num = pd.to_numeric(s, errors="coerce")
                if s_num.notna().any():
                    q50 = s_num.quantile(0.5)
                    mn = s_num.min()
                    mx = s_num.max()
                    sd = s_num.std() if s_num.count() > 1 else None
                    extra = f" | min={float(mn):.4g} | p50={float(q50):.4g} | max={float(mx):.4g}" + (f" | std={float(sd):.4g}" if sd is not None else "")
            else:
                # datetime heuristic
                if _looks_like_datetime(s):
                    parsed = _parse_datetime_series(s)
                    if parsed.notna().mean() > 0.5:
                        col_type = "datetime"
                        try:
                            extra = f" | min={str(parsed.min())} | max={str(parsed.max())}"
                        except Exception:
                            pass
                if col_type == "other":
                    # treat as category/text
                    s_str = s.astype("string")
                    col_type = "category"
                    try:
                        vc = s_str.value_counts(dropna=True)
                        if len(vc) > 0:
                            top_val = str(vc.index[0])
                            top_pct = float(vc.iloc[0] / max(1, nonnull)) * 100
                            extra = f" | top={top_val} ({top_pct:.1f}%)"
                    except Exception:
                        pass
                    # if very high cardinality, hint text-ish
                    try:
                        if unique > 0.8 * max(1, nonnull):
                            # average length as signal
                            avg_len = float(s_str.dropna().str.len().mean() or 0)
                            extra += f" | avg_len={avg_len:.1f}"
                            col_type = "text"
                    except Exception:
                        pass
        except Exception:
            pass

        flags_part = f" | flags={','.join(flags)}" if flags else ""
        line = (
            f"{col} | type={col_type} | nonnull={nonnull} | missing={missing_pct:.1f}% | unique={unique}" 
            f"{extra}{flags_part}"
        )
        lines.append(line)

    return lines


# --- Cross-column overview for Stage-2 prompts (very compact) ---
def build_cross_column_overview(df: pd.DataFrame) -> str:
    """Cross-column overview for Stage-2 prompts (very compact).
    Summarizes types, missingness, and key numeric/category signals across ALL columns.
    Returns a short multi-line string.
    """
    lines: list[str] = []
    n_rows = int(len(df))
    n_cols = int(df.shape[1])
    lines.append(f"Gesamt: {n_rows} Zeilen, {n_cols} Spalten")

    # --- Spaltentypen zählen ---
    try:
        num_cols = list(df.select_dtypes(include=["number"]).columns)
        obj_cols = list(df.select_dtypes(include=["object", "string", "category"]).columns)
        # Heuristik für Datums-Spalten (nicht zu teuer)
        dt_cols: list[str] = []
        for c in df.columns:
            try:
                if _looks_like_datetime(df[c]):
                    dt_cols.append(c)
            except Exception:
                continue
        # Text vs Kategorie grob trennen: hohe Kardinalität ~ Text
        cat_cols: list[str] = []
        text_like: list[str] = []
        for c in obj_cols:
            try:
                s = df[c].astype("string")
                uniq = int(s.nunique(dropna=True))
                nonnull = int(s.notna().sum())
                if nonnull > 0 and uniq > 0.8 * nonnull:
                    text_like.append(c)
                else:
                    cat_cols.append(c)
            except Exception:
                cat_cols.append(c)
        lines.append(
            f"Typen: numeric={len(num_cols)}, category={len(cat_cols)}, text≈{len(text_like)}, datetime≈{len(dt_cols)}"
        )
    except Exception:
        num_cols, cat_cols, text_like, dt_cols = [], [], [], []

    # --- Missingness (global + Top-5) ---
    try:
        miss_pct = df.isna().mean() * 100.0
        avg_missing = float(miss_pct.mean()) if len(miss_pct) else 0.0
        top_missing = miss_pct.sort_values(ascending=False).head(5)
        tm_str = ", ".join(f"{k}={v:.1f}%" for k, v in top_missing.items()) if not top_missing.empty else "–"
        lines.append(f"Fehlwerte: Ø {avg_missing:.1f}% | Top: {tm_str}")
    except Exception:
        lines.append("Fehlwerte: (n/a)")

    # --- Numeric: Verteilungsspread & Volatilität ---
    try:
        if num_cols:
            medians = {}
            stds = {}
            outlier_cols = 0
            outlier_ratio_mean = 0.0
            for c in num_cols:
                try:
                    s = pd.to_numeric(df[c], errors="coerce").dropna()
                    if s.empty:
                        continue
                    medians[c] = float(s.quantile(0.5))
                    stds[c] = float(s.std()) if s.size > 1 else 0.0
                    # IQR-Outlier Anteil
                    q1, q3 = s.quantile(0.25), s.quantile(0.75)
                    iqr = float(q3 - q1)
                    lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
                    m = (s < lower) | (s > upper)
                    out_c = int(m.sum())
                    if out_c > 0:
                        outlier_cols += 1
                    outlier_ratio_mean += (out_c / max(1, s.size))
                except Exception:
                    continue
            if medians:
                med_range = (min(medians.values()), max(medians.values()))
                std_sorted = sorted(stds.items(), key=lambda kv: (kv[1] if kv[1] is not None else 0.0), reverse=True)
                top_std = ", ".join(f"{k}={v:.1f}" for k, v in std_sorted[:5]) if std_sorted else "–"
                outlier_share = (outlier_cols / max(1, len(medians))) * 100.0
                outlier_ratio_mean = (outlier_ratio_mean / max(1, len(medians))) * 100.0
                lines.append(
                    f"Numerik: Median-Range={med_range[0]:.2f}–{med_range[1]:.2f} | Top Std: {top_std} | Outlier-Spalten≈{outlier_share:.1f}% (Ø Anteil {outlier_ratio_mean:.1f}%)"
                )
            else:
                lines.append("Numerik: (n/a)")
        else:
            lines.append("Numerik: (keine numerischen Spalten)")
    except Exception:
        lines.append("Numerik: (n/a)")

    # --- Kategorien: Kardinalität Top-5 ---
    try:
        if cat_cols:
            uniqs = []
            for c in cat_cols:
                try:
                    s = df[c].astype("string")
                    uniqs.append((c, int(s.nunique(dropna=True))))
                except Exception:
                    continue
            uniqs.sort(key=lambda kv: kv[1], reverse=True)
            top_uniq = ", ".join(f"{k}={v}" for k, v in uniqs[:5]) if uniqs else "–"
            lines.append(f"Kategorien: höchste Kardinalität: {top_uniq}")
        else:
            lines.append("Kategorien: (keine)")
    except Exception:
        lines.append("Kategorien: (n/a)")

    return "\n".join(lines)


from sqlalchemy import text as _sql_text

def get_dataset_original_name(engine, dataset_id: int) -> str:
    """Liefert den Original-Dateinamen zu dataset_id oder einen Fallback."""
    with engine.begin() as conn:
        row = conn.execute(
            _sql_text(
                """
                SELECT original_name
                FROM dataset_files
                WHERE dataset_id = :id
                """
            ),
            {"id": int(dataset_id)},
        ).mappings().first()
    return row["original_name"] if row else f"Dataset {int(dataset_id)}"


def build_dataset_context(engine, dataset_id: int, n_rows: int = 5, max_cols: int = 12, include_columns: list[str] | None = None) -> str:
    """Erzeuge einen kompakten Kontextblock aus dem gespeicherten CSV für LLM-Prompts.

    Inhalt:
      - Spaltenliste mit (vereinfachtem) dtype (max_cols begrenzt)
      - Bis zu n_rows Beispielzeilen
      - Basisstatistiken für numerische Spalten (count/mean/std/min/max, begrenzt)

    Der Kontext ist absichtlich kompakt, um Token zu sparen und Caching zu begünstigen.
    """
    # 1) Datei-Metadaten laden (Pfad, Encoding, Delimiter)
    with engine.begin() as conn:
        meta = conn.execute(
            _sql_text(
                """
                SELECT file_path, encoding, delimiter
                FROM dataset_files
                WHERE dataset_id = :id
                ORDER BY id DESC
                LIMIT 1
                """
            ),
            {"id": int(dataset_id)},
        ).mappings().first()

    if not meta:
        return (
            "(Kein Dateikontext gefunden: dataset_files-Eintrag fehlt. "
            "Bitte lade eine CSV und versuche es erneut.)"
        )

    file_path = meta["file_path"]
    preferred_enc = meta.get("encoding") if isinstance(meta, dict) else meta["encoding"]
    preferred_delim = meta.get("delimiter") if isinstance(meta, dict) else meta["delimiter"]

    # 2) CSV robust laden (gz-File)
    try:
        df, used_enc, used_delim = load_csv_resilient(
            file_path=file_path,
            preferred_encoding=preferred_enc,
            preferred_delimiter=preferred_delim,
        )
    except Exception as e:
        return f"(CSV konnte nicht geladen werden: {e})"

    # Optionally restrict to explicitly selected columns for Stage-2
    used_selected = False
    if include_columns:
        sel = [c for c in include_columns if c in df.columns]
        if sel:
            df = df[sel]
            used_selected = True

    # 3) Spaltenliste + dtypes (gekürzt)
    try:
        all_cols = list(map(str, df.columns))
        dtypes = [str(df[c].dtype) for c in df.columns]
        # If we are using an explicit selection, do not apply max_cols truncation
        if not used_selected and len(all_cols) > max_cols:
            cols_show = all_cols[:max_cols] + [f"… (+{len(all_cols)-max_cols} weitere)"]
            dtypes_show = dtypes[:max_cols] + ["…"]
        else:
            cols_show = all_cols
            dtypes_show = dtypes
        cols_block = "\n".join(f"- {c} ({t})" for c, t in zip(cols_show, dtypes_show))
        if used_selected:
            cols_block += "\n(Hinweis: Kontext auf ausgewählte Spalten beschränkt)"
    except Exception:
        cols_block = "(Spaltenliste nicht verfügbar)"

    # 4) Beispielzeilen (head)
    try:
        head_df = df.head(n_rows).copy()
        # String-Repräsentation kompakt halten
        example_block = head_df.to_csv(index=False)  # CSV-ähnlich ist für LLMs ok
    except Exception:
        example_block = "(Beispielzeilen nicht verfügbar)"

    # 5) Erweiterte Statistikblöcke: numerisch, Ausreißer, Extreme, Kategorisch, Histogramme
    #    Ziele:
    #      - Fünf-Zahlen-Zusammenfassung je numerischer Spalte
    #      - IQR-basierte Outlier-Zählung
    #      - Top/Bottom-k Extreme (Wert + Index)
    #      - Kategorische Top-N und Kardinalität
    #      - Leichte Histogramm-Bins (counts)
    try:
        # Limits, damit der Prompt kompakt bleibt
        num_limit = max(1, max_cols // 2)
        cat_limit = max(1, max_cols // 2)
        k_extremes = 3
        hist_bins = 10

        # --- Numerische Spalten: fünf Zahlen + mean/std
        num_df = df.select_dtypes(include=["number"]).copy()
        numeric_block = "(Keine numerischen Spalten)"
        outlier_block = "(Keine numerischen Spalten)"
        extremes_block = "(Keine numerischen Spalten)"
        hist_block = "(Keine numerischen Spalten)"
        if not num_df.empty:
            num_df = num_df.iloc[:, :num_limit]

            # Fünf-Zahlen + mean/std
            desc_rows = []
            for col in num_df.columns:
                s = pd.to_numeric(num_df[col], errors="coerce")
                if not s.notna().any():
                    continue
                q1 = s.quantile(0.25)
                q2 = s.quantile(0.50)
                q3 = s.quantile(0.75)
                row = {
                    "column": str(col),
                    "count": int(s.count()),
                    "mean": float(s.mean()),
                    "std": float(s.std()) if s.count() > 1 else None,
                    "min": float(s.min()),
                    "q25": float(q1),
                    "median": float(q2),
                    "q75": float(q3),
                    "max": float(s.max()),
                }
                desc_rows.append(row)
            if desc_rows:
                numeric_block = "column,count,mean,std,min,q25,median,q75,max\n" + "\n".join(
                    f"{r['column']},{r['count']},{round(r['mean'],4) if r['mean'] is not None else ''},"
                    f"{round(r['std'],4) if r['std'] is not None else ''},{round(r['min'],4)},"
                    f"{round(r['q25'],4)},{round(r['median'],4)},{round(r['q75'],4)},{round(r['max'],4)}"
                    for r in desc_rows
                )

            # IQR-Outlier je Spalte
            out_rows = []
            for col in num_df.columns:
                s = pd.to_numeric(num_df[col], errors="coerce")
                s = s.dropna()
                if s.empty:
                    continue
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = float(q3 - q1)
                lower = float(q1 - 1.5 * iqr)
                upper = float(q3 + 1.5 * iqr)
                mask = (s < lower) | (s > upper)
                out_count = int(mask.sum())
                out_ratio = float(out_count / max(1, s.size))
                out_rows.append({
                    "column": str(col),
                    "outliers": out_count,
                    "ratio": round(out_ratio, 6),
                    "lower": lower,
                    "upper": upper,
                })
            if out_rows:
                outlier_block = "column,outliers,ratio,lower,upper\n" + "\n".join(
                    f"{r['column']},{r['outliers']},{r['ratio']},{round(r['lower'],4)},{round(r['upper'],4)}"
                    for r in out_rows
                )

            # Extreme Werte (Top/Bottom k) – nur Wert + Index (kein ganzer Datensatz)
            ext_lines = ["column,type,index,value"]
            for col in num_df.columns:
                s = pd.to_numeric(num_df[col], errors="coerce")
                s = s.dropna()
                if s.empty:
                    continue
                # Bottom k
                bot = s.nsmallest(k_extremes)
                for idx, val in bot.items():
                    ext_lines.append(f"{col},min,{idx},{round(float(val),4)}")
                # Top k
                top = s.nlargest(k_extremes)
                for idx, val in top.items():
                    ext_lines.append(f"{col},max,{idx},{round(float(val),4)}")
            if len(ext_lines) > 1:
                extremes_block = "\n".join(ext_lines)

            # Histogramm-Bins (counts)
            hist_lines = ["column,bin_left,bin_right,count"]
            for col in num_df.columns:
                s = pd.to_numeric(num_df[col], errors="coerce")
                s = s.dropna()
                if s.empty:
                    continue
                try:
                    binned, edges = pd.cut(s, bins=hist_bins, retbins=True, include_lowest=True, duplicates="drop")
                    counts = binned.value_counts(sort=False)
                    # edges hat len = bins+1; counts hat len = bins_eff
                    for i, cnt in enumerate(counts.values):
                        left = float(edges[i])
                        right = float(edges[i+1])
                        hist_lines.append(f"{col},{round(left,4)},{round(right,4)},{int(cnt)}")
                except Exception:
                    continue
            if len(hist_lines) > 1:
                hist_block = "\n".join(hist_lines)

        # --- Kategorische Spalten
        cat_cols = [c for c in df.columns if str(df[c].dtype) not in map(str, df.select_dtypes(include=['number']).dtypes)]
        categorical_block = "(Keine kategorischen Spalten)"
        if cat_cols:
            # per dtype-Filter oben kommen ggf. zu viele; wir limitieren
            cat_cols = cat_cols[:cat_limit]
            lines = ["column,count,unique,top,freq"]
            for col in cat_cols:
                try:
                    s = df[col].astype("string")
                    if not s.notna().any():
                        continue
                    vc = s.value_counts(dropna=True)
                    top_val = str(vc.index[0]) if len(vc) > 0 else ""
                    top_freq = int(vc.iloc[0]) if len(vc) > 0 else 0
                    lines.append(f"{col},{int(s.count())},{int(s.nunique(dropna=True))},{top_val},{top_freq}")
                except Exception:
                    continue
            if len(lines) > 1:
                categorical_block = "\n".join(lines)
    except Exception:
        numeric_block = "(Statistiken nicht verfügbar)"
        outlier_block = "(Ausreißer-Berechnung nicht verfügbar)"
        extremes_block = "(Extreme nicht verfügbar)"
        hist_block = "(Histogramme nicht verfügbar)"
        categorical_block = "(Kategorische Zusammenfassung nicht verfügbar)"

    context = (
        "### Datensatz-Kontext\n"
        f"Spalten (max {max_cols}):\n{cols_block}\n\n"
        f"Beispielzeilen (n={n_rows}):\n{example_block}\n"
        f"Numerik (Fünf-Zahlen + mean/std):\n{numeric_block}\n\n"
        f"Ausreißer (IQR-Regel):\n{outlier_block}\n\n"
        f"Extreme Werte (Top/Bottom {k_extremes}):\n{extremes_block}\n\n"
        f"Histogramme (Counts, {hist_bins} Bins):\n{hist_block}\n\n"
        f"Kategorisch (Top-N & Kardinalität):\n{categorical_block}"
    )

    return context