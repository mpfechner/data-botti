import hashlib, gzip, io, os
from sqlalchemy import text
import pandas as pd

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
    """Robustes Einlesen einer gz-gepackten CSV-Datei mit Fallbacks für Encoding, Delimiter, Header,
    Dezimalkommas, NA-Varianten, duplizierte Headerzeilen, und wackelige Timestamps.

    Returns:
        df (pd.DataFrame), used_encoding (str), used_delimiter (str)
    """
    encodings = []
    if preferred_encoding:
        encodings.append(preferred_encoding)
    encodings += ["utf-8", "latin-1"]

    # Kandidatenreihenfolge für Delimiter
    base_delims = []
    if preferred_delimiter:
        base_delims.append(preferred_delimiter)
    base_delims += [",", ";"]

    last_err = None
    for enc in encodings:
        try:
            # Sample lesen für Sniffer
            with gzip.open(file_path, "rt", encoding=enc, newline="") as _f:
                sample = _f.read(4096)
        except Exception as e:
            last_err = e
            continue

        # 1) Delimiter-Kandidaten ausprobieren
        for delim in base_delims:
            try:
                with gzip.open(file_path, "rt", encoding=enc, newline="") as f:
                    df = pd.read_csv(
                        f,
                        delimiter=delim,
                        engine="python",
                        on_bad_lines="warn"
                    )
                used_delim = delim
                used_enc = enc
                break
            except Exception as e:
                last_err = e
                df = None
        if df is None:
            # 2) csv.Sniffer als Fallback
            try:
                dialect = csv.Sniffer().sniff(sample)
                sniffed_delim = dialect.delimiter
                with gzip.open(file_path, "rt", encoding=enc, newline="") as f:
                    df = pd.read_csv(
                        f,
                        delimiter=sniffed_delim,
                        engine="python",
                        on_bad_lines="warn"
                    )
                used_delim = sniffed_delim
                used_enc = enc
            except Exception as e:
                last_err = e
                df = None

        if df is not None:
            # ======= Cleanup-Phase =======
            # 1) Headerlos? -> generische Namen vergeben
            if all(isinstance(c, int) for c in df.columns):
                default_cols = [
                    "timestamp", "sensor_id", "temperature_c", "humidity_pct",
                    "battery_v", "status", "location", "note"
                ]
                df.columns = default_cols[: len(df.columns)]

            # 2) Duplizierte Headerzeilen erkennen & entfernen
            header_as_str = [str(c) for c in df.columns]
            try:
                mask_dupe_header = df.apply(lambda r: list(map(str, r.values.tolist())) == header_as_str, axis=1)
                if mask_dupe_header.any():
                    df = df.loc[~mask_dupe_header].copy()
            except Exception:
                pass

            # 3) Whitespace trimmen
            for col in df.select_dtypes(include=["object", "string"]).columns:
                try:
                    df[col] = df[col].astype("string").str.strip()
                except Exception:
                    pass

            # 4) NA-Tokens bereinigen
            na_tokens = {"NA": pd.NA, "NaN": pd.NA, "null": pd.NA, "": pd.NA}
            try:
                df.replace(na_tokens, inplace=True)
            except Exception:
                pass

            # 5) Dezimalkommas konvertieren
            import re
            decimal_pattern = re.compile(r"^\s*[-+]?\d{1,3}(?:[\d\.]*\d)?\,\d+\s*$")
            for col in df.columns:
                if df[col].dtype == object or str(df[col].dtype) == "string":
                    try:
                        frac = df[col].astype("string").str.fullmatch(decimal_pattern).mean()
                        if pd.notna(frac) and frac > 0.05:
                            df[col] = df[col].astype("string").str.replace(",", ".", regex=False)
                    except Exception:
                        pass

            # 6) Numerische Spalten coercen
            for col in ["temperature_c", "humidity_pct", "battery_v"]:
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    except Exception:
                        pass

            # 7) Timestamps robust parsen
            if "timestamp" in df.columns:
                try:
                    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=False, dayfirst=False)
                    if ts.isna().mean() > 0.2:
                        ts2 = pd.to_datetime(df["timestamp"], errors="coerce", utc=False, dayfirst=True)
                        ts = ts.fillna(ts2)
                    df["timestamp"] = ts
                except Exception:
                    pass

            # 8) Leere Zeilen entfernen
            try:
                df.dropna(how="all", inplace=True)
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
    for c in df.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                s = pd.to_datetime(df[c], errors="coerce")
            else:
                # Wenn nicht schon datetime: sanft versuchen (coerce)
                s = pd.to_datetime(df[c], errors="coerce", utc=False, dayfirst=False)
                if s.isna().mean() > 0.5:
                    # Wenn überwiegend NaT: lieber lassen
                    continue
            if s.notna().any():
                dt_info[c] = {"min": str(s.min()), "max": str(s.max())}
        except Exception:
            continue
    if dt_info:
        insights["datetime"] = dt_info

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