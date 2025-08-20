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

def insert_dataset_and_file(db_engine, user_id: int, name: str, file_info: dict) -> int:
    with db_engine.begin() as conn:
        dataset_id = conn.execute(
            text("""
                 INSERT INTO datasets (filename, upload_date, user_id)
                 VALUES (:filename, NOW(), :user_id)
                 """),
            {"user_id": user_id, "name": name}
        ).lastrowid

        conn.execute(
            text("""
                INSERT INTO dataset_files
                  (dataset_id, original_name, size_bytes, file_hash, encoding, delimiter, stored_at, file_path)
                VALUES
                  (:dataset_id, :original_name, :size_bytes, :file_hash, :encoding, :delimiter, NOW(), :file_path)
            """),
            {"dataset_id": dataset_id, **file_info}
        )
    return dataset_id


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