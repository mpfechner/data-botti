import hashlib, gzip, io, os
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

def sha256_bytesio(bio: io.BytesIO) -> str:
    bio.seek(0)
    h = hashlib.sha256()
    for chunk in iter(lambda: bio.read(1024 * 1024), b""):
        h.update(chunk)
    bio.seek(0)
    return h.hexdigest()

def save_gzip_to_data(bio: io.BytesIO, hexhash: str, data_dir: str = "data") -> str:
    bio.seek(0)
    out_path = os.path.join(data_dir, f"{hexhash}.csv.gz")
    with gzip.open(out_path, "wb") as gz:
        for chunk in iter(lambda: bio.read(1024 * 1024), b""):
            gz.write(chunk)
    return out_path

def insert_dataset_and_file(engine, user_id: int, filename: str, file_info: dict) -> int:
    """Legt Dataset + Datei an. Bei bereits vorhandenem file_hash wird der bestehende dataset_id zurückgegeben."""
    with engine.begin() as conn:
        # 1) Duplikate vermeiden – existiert der Hash schon?
        existing = conn.execute(
            text("SELECT dataset_id FROM dataset_files WHERE file_hash = :h LIMIT 1"),
            {"h": file_info["file_hash"]}
        ).scalar()
        if existing:
            return int(existing)

        # 2) Neues Dataset anlegen
        res = conn.execute(
            text("""
                INSERT INTO datasets (filename, upload_date, user_id)
                VALUES (:filename, NOW(), :user_id)
            """),
            {"filename": filename, "user_id": user_id}
        )
        dataset_id = getattr(res, "lastrowid", None)
        if not dataset_id:
            dataset_id = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()

        # 3) Datei-Metadaten anlegen
        conn.execute(
            text("""
                INSERT INTO dataset_files (dataset_id, original_name, size_bytes, file_hash, encoding, delimiter, file_path)
                VALUES (:dataset_id, :original_name, :size_bytes, :file_hash, :encoding, :delimiter, :file_path)
            """),
            {
                "dataset_id": dataset_id,
                "original_name": file_info["original_name"],
                "size_bytes": file_info["size_bytes"],
                "file_hash": file_info["file_hash"],
                "encoding": file_info["encoding"],
                "delimiter": file_info["delimiter"],
                "file_path": file_info["file_path"],
            }
        )

    return int(dataset_id)


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


def load_csv_resilient(file_path: str, preferred_encoding: str | None = None, preferred_delimiter: str | None = None):
    """Robustes Einlesen einer gz-gepackten CSV-Datei mit Fallbacks:
    - Encoding: preferred → utf-8 → latin-1
    - Delimiter: preferred → ',', ';', '\\t', '|' → csv.Sniffer
    - Header-Erkennung: vergleicht die ersten ZWEI Zeilen; wenn 1. Zeile wie Daten aussieht → header=None + generische Spaltennamen
    - Cleanup: NA-Tokens, Whitespace, Dezimal-Kommas, duplizierte Headerzeilen
    Returns:
        df (pd.DataFrame), used_encoding (str), used_delimiter (str)
    """
    import csv, re

    encodings = [e for e in [preferred_encoding, "utf-8", "latin-1"] if e]
    delim_candidates = [d for d in [preferred_delimiter, ",", ";", "\t", "|"] if d]

    last_err = None
    for enc in encodings:
        # Sample lesen (für Sniffer & Heuristiken)
        try:
            with gzip.open(file_path, "rt", encoding=enc, newline="") as _f:
                sample = _f.read(8192)
        except Exception as e:
            last_err = e
            continue

        # --- Delimiter bestimmen ---
        used_delim = None
        for delim in delim_candidates:
            try:
                with gzip.open(file_path, "rt", encoding=enc, newline="") as f_test:
                    test_df = pd.read_csv(f_test, delimiter=delim, engine="python", on_bad_lines="warn", nrows=50, header=0)
                if test_df.shape[1] > 1:
                    used_delim = delim
                    break
            except Exception as e:
                last_err = e
                continue
        if used_delim is None:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
                used_delim = dialect.delimiter
            except Exception:
                used_delim = ","

        # ---- Erste ZWEI Zeilen tokenisieren für Header-Heuristik
        def _tokenize_first_lines(delim: str):
            try:
                with gzip.open(file_path, "rt", encoding=enc, newline="") as f:
                    l1 = f.readline().rstrip("\n")
                    l2 = f.readline().rstrip("\n")
                r1 = next(csv.reader([l1], delimiter=delim))
                r2 = next(csv.reader([l2], delimiter=delim)) if l2 else []
                return r1, r2
            except Exception:
                return [], []

        first_tokens, second_tokens = _tokenize_first_lines(used_delim)

        num_re = re.compile(r"^\s*[-+]?\d+(?:[.,]\d+)?\s*$")
        time_re = re.compile(r"\d{1,2}:\d{2}")
        dateish_re = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")

        def _tok_type(s: str) -> str:
            s = (s or "").strip()
            if not s:
                return "empty"
            if num_re.fullmatch(s):
                return "num"
            if time_re.search(s) or dateish_re.search(s):
                return "dt"
            return "txt"

        def _looks_like_header(toks1: list[str], toks2: list[str]) -> bool:
            """Stärkerer Test:
            - Wenn >50% der Tokens in Zeile 1 num/datetime → eher DATEN.
            - Wenn 1. und 2. Zeile im Token-Typ-Muster stark übereinstimmen → eher DATEN.
            - Sonst Header.
            """
            if not toks1:
                return True  # lieber Header annehmen, falls wir nichts wissen

            types1 = [_tok_type(t) for t in toks1]
            frac_data = sum(t in ("num", "dt") for t in types1) / max(1, len(types1))
            if frac_data > 0.5:
                return False  # eher Daten

            if toks2 and len(toks2) == len(toks1):
                types2 = [_tok_type(t) for t in toks2]
                equal_types = sum(a == b for a, b in zip(types1, types2)) / max(1, len(types1))
                if equal_types >= 0.6:
                    return False  # zwei inhaltlich ähnliche Zeilen → Daten

            # Header-Indiz: mehrere „namenähnliche“ Tokens ohne Zahlen
            headerish = sum(t == "txt" for t in types1) / max(1, len(types1)) >= 0.6
            return headerish

        header_is_ok = _looks_like_header(first_tokens, second_tokens)

        # --- Datei einlesen (mit/ohne Header)
        try:
            if header_is_ok:
                with gzip.open(file_path, "rt", encoding=enc, newline="") as f_full:
                    df = pd.read_csv(f_full, delimiter=used_delim, engine="python", on_bad_lines="warn", header="infer")
                header_detected = True
            else:
                with gzip.open(file_path, "rt", encoding=enc, newline="") as f_full:
                    df = pd.read_csv(f_full, delimiter=used_delim, engine="python", on_bad_lines="warn", header=None)
                df.columns = [f"col_{i}" for i in range(df.shape[1])]
                header_detected = False
            used_enc = enc
        except Exception as e:
            last_err = e
            continue

        # ======= Cleanup-Phase (generisch) =======
        header_as_str = [str(c) for c in df.columns]
        try:
            mask_dupe_header = df.apply(lambda r: list(map(str, r.values.tolist())) == header_as_str, axis=1)
            if mask_dupe_header.any():
                df = df.loc[~mask_dupe_header].copy()
        except Exception:
            pass

        for col in df.select_dtypes(include=["object", "string"]).columns:
            try:
                df[col] = df[col].astype("string").str.strip()
            except Exception:
                pass

        try:
            df.replace({"NA": pd.NA, "NaN": pd.NA, "null": pd.NA, "": pd.NA}, inplace=True)
        except Exception:
            pass

        # Dezimalkommas → Punkte (spaltenweise Heuristik)
        decimal_pattern = re.compile(r"^\s*[-+]?\d{1,3}(?:[\d\.]*\d)?\,\d+\s*$")
        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype) == "string":
                try:
                    frac = df[col].astype("string").str.fullmatch(decimal_pattern).mean()
                    if pd.notna(frac) and frac > 0.05:
                        df[col] = df[col].astype("string").str.replace(",", ".", regex=False)
                        # sanft numerisch konvertieren (coerce), aber nur übernehmen, wenn überwiegend numerisch
                        conv = pd.to_numeric(df[col], errors="coerce")
                        if conv.notna().mean() > 0.5:
                            df[col] = conv
                except Exception:
                    pass

        try:
            df.dropna(how="all", inplace=True)
        except Exception:
            pass

        # Flag für nachgelagerte UI
        try:
            df.attrs["header_detected"] = bool(header_detected)
        except Exception:
            pass

        return df, used_enc, used_delim

    raise RuntimeError(f"CSV konnte nicht gelesen werden. Letzter Fehler: {last_err}")


def get_or_create_default_user(engine) -> int:
    """Gibt eine gültige users.id zurück; legt bei Bedarf einen Default-User an.
    Erwartete Tabellenspalten: id (PK, auto_inc), username (TEXT), email (TEXT).
    """
    with engine.begin() as conn:
        uid = conn.execute(text("SELECT id FROM users ORDER BY id LIMIT 1")).scalar()
        if uid:
            return int(uid)

        res = conn.execute(
            text("""
                INSERT INTO users (username, email)
                VALUES (:u, :e)
            """),
            {"u": "databotti_default", "e": "default@example.com"}
        )
        new_id = getattr(res, "lastrowid", None)
        if new_id is None:
            new_id = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()
        return int(new_id)


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