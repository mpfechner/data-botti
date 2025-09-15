from __future__ import annotations
from typing import Optional, Mapping
from sqlalchemy import text


# --- dataset_files repository helpers -----------------------------------------------------------

def get_latest_dataset_file(conn, dataset_id: int) -> Optional[Mapping]:
    """Return latest dataset_files row for a given dataset_id as a mapping.

    Expected columns: file_path, encoding, delimiter, original_name, file_hash. Returns None if not found.
    Usage:
        with engine.begin() as conn:
            meta = get_latest_dataset_file(conn, dataset_id)
    """
    return conn.execute(
        text(
            """
            SELECT file_path, encoding, delimiter, original_name, file_hash
            FROM dataset_files
            WHERE dataset_id = :id
            ORDER BY id DESC
            LIMIT 1
            """
        ),
        {"id": int(dataset_id)},
    ).mappings().first()


def add_dataset_file(
    conn,
    dataset_id: int,
    file_path: str,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> None:
    """Insert a new dataset_files row.

    Usage:
        with engine.begin() as conn:
            add_dataset_file(conn, ds_id, path, encoding, delimiter)
    """
    conn.execute(
        text(
            """
            INSERT INTO dataset_files (dataset_id, file_path, encoding, delimiter)
            VALUES (:dataset_id, :file_path, :encoding, :delimiter)
            """
        ),
        {
            "dataset_id": int(dataset_id),
            "file_path": file_path,
            "encoding": encoding,
            "delimiter": delimiter,
        },
    )


def get_dataset_file_meta(conn, dataset_id: int) -> Optional[Mapping]:
    """Return one dataset_files row for a dataset_id (no explicit ordering)."""
    return conn.execute(
        text(
            """
            SELECT file_path, encoding, delimiter, original_name, file_hash
            FROM dataset_files
            WHERE dataset_id = :id
            """
        ),
        {"id": int(dataset_id)},
    ).mappings().first()


# --- dataset_columns repository helpers ---------------------------------------------------------

def get_dataset_columns(conn, dataset_id: int):
    """Return ordered column metadata rows from dataset_columns for a dataset_id."""
    return conn.execute(
        text(
            """
            SELECT ordinal, name, dtype, is_nullable, distinct_count, min_val, max_val
            FROM dataset_columns
            WHERE dataset_id = :id
            ORDER BY ordinal
            """
        ),
        {"id": int(dataset_id)},
    ).mappings().all()


# --- Additional dataset_files helpers ----------------------------------------------------------

def get_dataset_files(conn, dataset_id: int):
    """Return all dataset_files rows for a dataset_id ordered by newest first.
    Columns returned: id, file_path, encoding, delimiter, original_name, file_hash
    """
    return conn.execute(
        text(
            """
            SELECT id, file_path, encoding, delimiter, original_name, file_hash
            FROM dataset_files
            WHERE dataset_id = :id
            ORDER BY id DESC
            """
        ),
        {"id": int(dataset_id)},
    ).mappings().all()


# --- Helper to insert dataset and its initial file in a single transaction ---------------------

def insert_dataset_and_file(
    conn,
    user_id: int,
    original_name: str,
    file_hash: str,
    file_path: str,
    size_bytes: int | None = None,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> tuple[int, bool]:
    """Create a new dataset and its first dataset_files row. Returns (dataset_id, is_new).

    This mirrors the previous helper in helpers.py but centralizes SQL in the repo layer.
    It assumes a MySQL backend with AUTO_INCREMENT on datasets.id.
    size_bytes is also stored in dataset_files.
    """
    # Deduplicate by file_hash: if this file already exists, reuse its dataset_id and skip inserts
    existing = conn.execute(
        text(
            """
            SELECT dataset_id
            FROM dataset_files
            WHERE file_hash = :file_hash
            LIMIT 1
            """
        ),
        {"file_hash": file_hash},
    ).mappings().first()
    if existing and existing.get("dataset_id"):
        return int(existing["dataset_id"]), False  # Reuse existing dataset; unique constraint enforces one owner per file

    # Insert dataset row
    res = conn.execute(
        text(
            """
            INSERT INTO datasets (user_id, filename, upload_date)
            VALUES (:user_id, :filename, NOW())
            """
        ),
        {"user_id": int(user_id), "filename": original_name},
    )

    # Try to retrieve the new primary key
    dataset_id = None
    try:
        dataset_id = int(res.lastrowid) if getattr(res, "lastrowid", None) is not None else None
    except Exception:
        dataset_id = None
    if not dataset_id:
        try:
            dataset_id = int(
                conn.execute(text("SELECT LAST_INSERT_ID() AS id")).scalar()
            )
        except Exception:
            dataset_id = None
    if dataset_id is None:
        raise RuntimeError("Konnte dataset_id nach INSERT nicht bestimmen")

    # Insert dataset_files row
    conn.execute(
        text(
            """
            INSERT INTO dataset_files (dataset_id, file_path, encoding, delimiter, original_name, file_hash, size_bytes)
            VALUES (:dataset_id, :file_path, :encoding, :delimiter, :original_name, :file_hash, :size_bytes)
            """
        ),
        {
            "dataset_id": dataset_id,
            "file_path": file_path,
            "encoding": encoding,
            "delimiter": delimiter,
            "original_name": original_name,
            "file_hash": file_hash,
            "size_bytes": size_bytes,
        },
    )

    return dataset_id, True