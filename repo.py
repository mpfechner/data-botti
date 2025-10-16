from __future__ import annotations
from db import get_engine
from typing import Optional, Mapping
from sqlalchemy import text
from flask import current_app
import json

import pandas as pd
from services.csv_io import load_csv_resilient
from sqlalchemy.exc import IntegrityError
from services.models import QARecord

# --- get_dataset_original_name: in-repo implementation (replaces legacy helper) ----------------

def get_dataset_original_name(engine, dataset_id: int) -> str:
    """Return the original file name for a dataset.

    Prefers the latest entry in dataset_files.original_name; falls back to datasets.filename.
    Returns an empty string if neither is found.
    """
    with engine.connect() as conn:
        # Prefer most recent dataset_files.original_name
        orig = conn.execute(
            text(
                """
                SELECT original_name
                FROM dataset_files
                WHERE dataset_id = :id
                ORDER BY id DESC
                LIMIT 1
                """
            ),
            {"id": int(dataset_id)},
        ).scalar()
        if orig:
            return str(orig)

        # Fallback to datasets.filename
        name = conn.execute(
            text("SELECT filename FROM datasets WHERE id = :id"),
            {"id": int(dataset_id)},
        ).scalar()
        return str(name) if name is not None else ""



# --- build_dataset_context: in-repo implementation (replaces legacy helper) --------------------

def build_dataset_context(
    engine,
    dataset_id: int,
    n_rows: int = 5,
    max_cols: int = 12,
    include_columns: list[str] | None = None,
) -> str:
    """Build a compact textual context for a dataset.

    - Loads the latest dataset file (encoding/delimiter from dataset_files if available)
    - Optionally restricts to include_columns
    - Otherwise limits to the first max_cols columns
    - Returns a string with filename, column list and a small CSV preview (head n_rows)
    """
    # Fetch meta + load DataFrame
    with engine.begin() as conn:
        meta = get_latest_dataset_file(conn, dataset_id)
    original_name = get_dataset_original_name(engine, dataset_id)

    df = None
    used_encoding = None
    used_delimiter = None

    if meta and meta.get("file_path"):
        enc = meta.get("encoding")
        delim = meta.get("delimiter")
        # load_csv_resilient returns (df, used_encoding, used_delimiter)
        df, used_encoding, used_delimiter = load_csv_resilient(meta["file_path"], enc, delim)
    else:
        # Meta missing: try to load anyway (will likely fail but keeps behavior explicit)
        raise RuntimeError("No dataset_files metadata found for dataset; cannot build context")

    # Column selection logic
    all_cols = list(df.columns)
    if include_columns:
        cols = [c for c in include_columns if c in all_cols]
        used_selected = True
    else:
        used_selected = False
        cols = all_cols[: max_cols] if len(all_cols) > max_cols else all_cols

    # Compose preview
    preview_df = df[cols].head(int(n_rows)) if cols else df.head(int(n_rows))
    csv_preview = preview_df.to_csv(index=False)

    # Build context string
    header_lines = []
    if original_name:
        header_lines.append(f"Dataset: {original_name}")
    header_lines.append(f"Rows (preview): {int(n_rows)}  |  Columns used: {len(cols)} / {len(all_cols)}")
    header_lines.append("Columns: " + ", ".join(map(str, cols)))
    if used_encoding or used_delimiter:
        parts = []
        if used_encoding:
            parts.append(f"encoding={used_encoding}")
        if used_delimiter:
            parts.append(f"delimiter={used_delimiter}")
        header_lines.append("Load params: " + ", ".join(parts))

    context = "\n".join(header_lines) + "\n\n" + csv_preview
    return context


class DuplicateEmailError(Exception):
    pass

class DuplicateGroupNameError(Exception):
    pass


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


# --- analyze_and_store_columns: in-repo implementation (replaces legacy helper) -----------------

def analyze_and_store_columns(db_engine, dataset_id: int, df):
    """Analyze DataFrame columns and persist results to dataset_columns.

    Columns written: dataset_id, ordinal, name, dtype, is_nullable, distinct_count, min_val, max_val
    """
    # Build per-column metadata
    rows = []
    for idx, col in enumerate(df.columns):
        s = df[col]
        dtype_str = str(s.dtype)
        is_nullable = bool(getattr(s, "isna", lambda: pd.Series([False]))().any())
        # distinct
        try:
            distinct_count = int(s.nunique(dropna=True))
        except Exception:
            distinct_count = None

        min_val = None
        max_val = None

        if pd.api.types.is_numeric_dtype(s):
            try:
                num = pd.to_numeric(s, errors="coerce")
                if num.notna().any():
                    min_val = float(num.min())
                    max_val = float(num.max())
            except Exception:
                min_val = None
                max_val = None
        elif pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_datetime64_dtype(s):
            try:
                dt = pd.to_datetime(s, errors="coerce")
                if dt.notna().any():
                    min_val = dt.min().isoformat()
                    max_val = dt.max().isoformat()
                dtype_str = "datetime"
            except Exception:
                dtype_str = "datetime"
                min_val = None
                max_val = None
        else:
            # Treat object/string-like columns as string for downstream usage
            if pd.api.types.is_string_dtype(s) or s.dtype == object:
                dtype_str = "string"

        rows.append(
            {
                "ordinal": int(idx),
                "name": str(col),
                "dtype": dtype_str,
                "is_nullable": 1 if is_nullable else 0,
                "distinct_count": distinct_count,
                "min_val": min_val,
                "max_val": max_val,
            }
        )

    # Persist in a single transaction
    with db_engine.begin() as conn:
        conn.execute(text("DELETE FROM dataset_columns WHERE dataset_id = :id"), {"id": int(dataset_id)})
        for r in rows:
            conn.execute(
                text(
                    """
                    INSERT INTO dataset_columns
                        (dataset_id, ordinal, name, dtype, is_nullable, distinct_count, min_val, max_val)
                    VALUES
                        (:dataset_id, :ordinal, :name, :dtype, :is_nullable, :distinct_count, :min_val, :max_val)
                    """
                ),
                {
                    "dataset_id": int(dataset_id),
                    "ordinal": r["ordinal"],
                    "name": r["name"],
                    "dtype": r["dtype"],
                    "is_nullable": r["is_nullable"],
                    "distinct_count": r["distinct_count"],
                    "min_val": r["min_val"],
                    "max_val": r["max_val"],
                },
            )


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


# --- user repository helpers (used by routes/admin.py) -----------------------------------------



def repo_list_users():
    """Return all users with id, email, username, is_admin."""
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, email, username, is_admin
                FROM users
                ORDER BY id
                """
            )
        ).mappings().all()
    return rows


def repo_create_user(*, email: str, username: str | None, password_hash: str, is_admin: int = 0) -> int:
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.begin() as conn:
        try:
            res = conn.execute(
                text(
                    """
                    INSERT INTO users (email, username, password_hash, is_admin)
                    VALUES (:email, :username, :password_hash, :is_admin)
                    """
                ),
                {
                    "email": email,
                    "username": username,
                    "password_hash": password_hash,
                    "is_admin": int(bool(is_admin)),
                },
            )
        except IntegrityError as e:
            # Map DB duplicate error to domain-specific
            raise DuplicateEmailError() from e
        new_id = None
        try:
            new_id = int(getattr(res, "lastrowid", None) or 0) or None
        except Exception:
            new_id = None
        if new_id is None:
            try:
                new_id = int(conn.execute(text("SELECT LAST_INSERT_ID() AS id")).scalar())
            except Exception:
                new_id = None
        if new_id is None:
            raise RuntimeError("Could not determine new user id after insert")
        return new_id


def repo_update_user_password(*, user_id: int, password_hash: str) -> None:
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE users SET password_hash = :password_hash
                WHERE id = :user_id
                """
            ),
            {"password_hash": password_hash, "user_id": int(user_id)},
        )


def repo_delete_user(*, user_id: int) -> None:
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.begin() as conn:
        conn.execute(text("DELETE FROM users WHERE id = :user_id"), {"user_id": int(user_id)})


# --- group & membership repository helpers -----------------------------------------------------

def repo_list_groups():
    """Return all groups with member_count.
    Columns: id, name, created_at, member_count
    """
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT g.id, g.name, g.created_at,
                       (SELECT COUNT(*) FROM user_groups ug WHERE ug.group_id = g.id) AS member_count
                FROM groups g
                ORDER BY g.name
                """
            )
        ).mappings().all()
    return rows


essential_dupe_sql = """
INSERT INTO user_groups (user_id, group_id)
VALUES (:user_id, :group_id)
ON DUPLICATE KEY UPDATE group_id = group_id
"""


def repo_create_group(*, name: str) -> int:
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.begin() as conn:
        try:
            res = conn.execute(text("INSERT INTO groups (name) VALUES (:name)"), {"name": name})
        except IntegrityError as e:
            raise DuplicateGroupNameError() from e
        new_id = None
        try:
            new_id = int(getattr(res, "lastrowid", None) or 0) or None
        except Exception:
            new_id = None
        if new_id is None:
            try:
                new_id = int(conn.execute(text("SELECT LAST_INSERT_ID() AS id")).scalar())
            except Exception:
                new_id = None
        if new_id is None:
            raise RuntimeError("Could not determine new group id after insert")
        return new_id


def repo_delete_group(*, group_id: int) -> None:
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.begin() as conn:
        conn.execute(text("DELETE FROM groups WHERE id = :id"), {"id": int(group_id)})


def repo_list_group_members(*, group_id: int):
    """Return members of a group with user info.
    Columns: id, email, username, is_admin, added_at
    """
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT u.id, u.email, u.username, u.is_admin, ug.added_at
                FROM user_groups ug
                JOIN users u ON u.id = ug.user_id
                WHERE ug.group_id = :gid
                ORDER BY u.email
                """
            ),
            {"gid": int(group_id)},
        ).mappings().all()
    return rows


def repo_add_user_to_group(*, user_id: int, group_id: int) -> None:
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.begin() as conn:
        try:
            # MySQL-friendly upsert to ignore duplicates
            conn.execute(
                text(essential_dupe_sql),
                {"user_id": int(user_id), "group_id": int(group_id)},
            )
        except IntegrityError:
            # Fallback for strict modes or other engines: ignore if already exists
            pass


def repo_remove_user_from_group(*, user_id: int, group_id: int) -> None:
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.begin() as conn:
        conn.execute(
            text("DELETE FROM user_groups WHERE user_id = :uid AND group_id = :gid"),
            {"uid": int(user_id), "gid": int(group_id)},
        )

# --- admin guardrail helpers -------------------------------------------------------------------

def repo_is_user_admin(*, user_id: int) -> bool:
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.connect() as conn:
        val = conn.execute(text("SELECT is_admin FROM users WHERE id = :id"), {"id": int(user_id)}).scalar()
    try:
        return bool(int(val)) if val is not None else False
    except Exception:
        return False


def repo_count_admins() -> int:
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM users WHERE is_admin = 1")).scalar()
    try:
        return int(n) if n is not None else 0
    except Exception:
        return 0

# --- qa_pairs repository helpers ---------------------------------------------------------------

def _row_to_qarecord(row: Mapping | None) -> QARecord | None:
    """Map a qa_pairs row mapping to a QARecord dataclass. Returns None if row is None."""
    if row is None:
        return None
    # qa_pairs columns: id, file_hash, question_original, question_norm, question_hash, answer, meta, created_at
    return QARecord(
        id=int(row.get("id")) if row.get("id") is not None else None,
        question=row.get("question_original") or "",
        question_norm=row.get("question_norm") or "",
        question_hash=row.get("question_hash") or "",
        answer=row.get("answer"),
        file_hash=row.get("file_hash"),
        embedding=None,
        embed_model=None,
        created_at=row.get("created_at"),
        updated_at=row.get("created_at"),  # fallback when updated_at not present
    )

def _to_db_json(value):
    """Return a JSON-encoded string if value is a dict/list, else return as-is.
    MySQL/PyMySQL cannot bind Python dicts directly; JSON columns expect a string.
    """
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value

def repo_qa_insert(
    *,
    file_hash: str,
    question_original: str,
    question_norm: str,
    question_hash: str,
    answer: str | None = None,
    meta: dict | None = None,
) -> int:
    """Insert new qa_pairs row and return its id."""
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.begin() as conn:
        res = conn.execute(
            text(
                """
                INSERT INTO qa_pairs (file_hash, question_original, question_norm, question_hash, answer, meta)
                VALUES (:file_hash, :question_original, :question_norm, :question_hash, :answer, :meta)
                """
            ),
            {
                "file_hash": file_hash,
                "question_original": question_original,
                "question_norm": question_norm,
                "question_hash": question_hash,
                "answer": answer,
                "meta": _to_db_json(meta),
            },
        )
        new_id = None
        try:
            new_id = int(getattr(res, "lastrowid", None) or 0) or None
        except Exception:
            new_id = None
        if new_id is None:
            try:
                new_id = int(conn.execute(text("SELECT LAST_INSERT_ID() AS id")).scalar())
            except Exception:
                new_id = None
        if new_id is None:
            raise RuntimeError("Could not determine new qa_pairs id after insert")
        return new_id


def repo_qa_find_by_hash(*, file_hash: str, question_hash: str) -> QARecord | None:
    """Find a qa_pairs row by file_hash + question_hash. Returns QARecord or None."""
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT id, file_hash, question_original, question_norm, question_hash, answer, meta, created_at
                FROM qa_pairs
                WHERE file_hash = :file_hash AND question_hash = :question_hash
                LIMIT 1
                """
            ),
            {"file_hash": file_hash, "question_hash": question_hash},
        ).mappings().first()
    return _row_to_qarecord(row)

# New: Find a qa_pairs row by id
def repo_qa_find_by_id(*, qa_id: int) -> QARecord | None:
    """Find a qa_pairs row by id. Returns QARecord or None."""
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT id, file_hash, question_original, question_norm, question_hash, answer, meta, created_at
                FROM qa_pairs
                WHERE id = :id
                LIMIT 1
                """
            ),
            {"id": int(qa_id)},
        ).mappings().first()
    return _row_to_qarecord(row)


# --- qa_embeddings repository helpers ----------------------------------------------------------

def repo_qa_save_embedding(*, qa_id: int, model: str, dim: int, vec: bytes) -> None:
    """Insert or update an embedding vector for a given QA id and model."""
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO qa_embeddings (qa_id, model, dim, vec)
                VALUES (:qa_id, :model, :dim, :vec)
                ON DUPLICATE KEY UPDATE
                    dim = VALUES(dim),
                    vec = VALUES(vec),
                    created_at = CURRENT_TIMESTAMP
                """
            ),
            {"qa_id": qa_id, "model": model, "dim": dim, "vec": vec},
        )


def repo_embeddings_by_file(*, file_hash: str, model: str):
    """Return list of embeddings (qa_id, dim, vec bytes) for a given file_hash and model."""
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT qe.qa_id AS qa_id, qe.dim AS dim, qe.vec AS vec
                FROM qa_embeddings qe
                JOIN qa_pairs qp ON qp.id = qe.qa_id
                WHERE qp.file_hash = :file_hash AND qe.model = :model
                """
            ),
            {"file_hash": file_hash, "model": model},
        ).fetchall()
    return rows


# --- test/maintenance cleanup helpers ---------------------------------------------------------



# QA-scoped embedding cleanup helper
def repo_embeddings_delete_by_qa(*, qa_id: int, model: str) -> int:
    """Delete embeddings for a specific QA and model. Returns affected row count."""
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.begin() as conn:
        res = conn.execute(
            text("DELETE FROM qa_embeddings WHERE qa_id = :qa_id AND model = :model"),
            {"qa_id": int(qa_id), "model": model},
        )
        try:
            return int(getattr(res, "rowcount", 0) or 0)
        except Exception:
            return 0


def repo_qa_delete_by_id(*, qa_id: int) -> None:
    """Delete a qa_pairs row by id. Call after removing dependent embeddings."""
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.begin() as conn:
        conn.execute(text("DELETE FROM qa_pairs WHERE id = :id"), {"id": int(qa_id)})


# --- helper: list qa_pairs without embedding for a model --------------------------------------

MODEL_NAME = "intfloat/multilingual-e5-base"  # fallback default if not present elsewhere
GLOBAL_SEED_FILE_HASH = "seed_data"

def repo_qa_without_embedding(*, file_hash: str | None = None, model: str = MODEL_NAME):
    """Return list of qa_pairs (id, file_hash, question_norm) that have no embedding stored for the given model.
    If file_hash is provided, filter by that file; else return across all files.
    """
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.connect() as conn:
        if file_hash:
            rows = conn.execute(
                text(
                    """
                    SELECT qp.id, qp.file_hash, qp.question_norm
                    FROM qa_pairs qp
                    WHERE qp.file_hash = :file_hash
                      AND NOT EXISTS (
                          SELECT 1 FROM qa_embeddings qe
                          WHERE qe.qa_id = qp.id AND qe.model = :model
                      )
                    ORDER BY qp.id
                    """
                ),
                {"file_hash": file_hash, "model": model},
            ).mappings().all()
        else:
            rows = conn.execute(
                text(
                    """
                    SELECT qp.id, qp.file_hash, qp.question_norm
                    FROM qa_pairs qp
                    WHERE NOT EXISTS (
                          SELECT 1 FROM qa_embeddings qe
                          WHERE qe.qa_id = qp.id AND qe.model = :model
                      )
                    ORDER BY qp.id
                    """
                ),
                {"model": model},
            ).mappings().all()
    return rows


# --- semantic candidates helper ---------------------------------------------------------------

def repo_qa_semantic_candidates(*, file_hash: str, model: str):
    """Return list of (QARecord, dim, vec_bytes) for given file_hash and embedding model.

    Joins qa_pairs with qa_embeddings to fetch the question/answer context + stored embedding vector.
    The vector is returned as raw bytes; the caller is responsible for decoding to numpy array.
    """
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    qp.id, qp.file_hash, qp.question_original, qp.question_norm,
                    qp.question_hash, qp.answer, qp.meta, qp.created_at,
                    qe.dim AS dim, qe.vec AS vec
                FROM qa_pairs qp
                JOIN qa_embeddings qe ON qe.qa_id = qp.id
                WHERE qp.file_hash = :file_hash AND qe.model = :model
                ORDER BY qp.id DESC
                """
            ),
            {"file_hash": file_hash, "model": model},
        ).mappings().all()

    result = []
    for row in rows:
        rec = _row_to_qarecord(row)
        result.append((rec, int(row["dim"]) if row.get("dim") is not None else None, row.get("vec")))
    return result


def repo_qa_candidates_for_dataset(*, dataset_id: int, model: str, limit: int = 50):
    """Return list of (QARecord, dim, vec_bytes) for a given dataset_id and embedding model.

    This joins `dataset_files` (to resolve the dataset's file_hash) with `qa_pairs` and `qa_embeddings`.
    The vector is returned as raw bytes; the caller is responsible for decoding to a numpy array.
    Results are ordered by newest question first and limited by `limit`.
    """
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    qp.id, qp.file_hash, qp.question_original, qp.question_norm,
                    qp.question_hash, qp.answer, qp.meta, qp.created_at,
                    qe.dim AS dim, qe.vec AS vec
                FROM dataset_files df
                JOIN qa_pairs qp ON qp.file_hash = df.file_hash
                JOIN qa_embeddings qe ON qe.qa_id = qp.id AND qe.model = :model
                WHERE df.dataset_id = :dsid
                ORDER BY qp.created_at DESC
                LIMIT :limit
                """
            ),
            {"dsid": int(dataset_id), "model": model, "limit": int(limit)},
        ).mappings().all()

    result = []
    for row in rows:
        rec = _row_to_qarecord(row)
        result.append((rec, int(row["dim"]) if row.get("dim") is not None else None, row.get("vec")))
    return result


# New: Return QA candidates for dataset, optionally including global seed QA pairs
def repo_qa_candidates_for_dataset_with_seed(*, dataset_id: int, model: str, include_seed: bool = True, limit: int = 100):
    """Return list of (QARecord, dim, vec_bytes) for a given dataset_id and embedding model,
    optionally UNIONed with global seed QA pairs (file_hash == GLOBAL_SEED_FILE_HASH).

    Results are ordered by newest question first and limited by `limit`.
    """
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")

    if include_seed:
        sql = text(
            """
            (
                SELECT
                    qp.id, qp.file_hash, qp.question_original, qp.question_norm,
                    qp.question_hash, qp.answer, qp.meta, qp.created_at,
                    qe.dim AS dim, qe.vec AS vec
                FROM dataset_files df
                JOIN qa_pairs qp ON qp.file_hash = df.file_hash
                JOIN qa_embeddings qe ON qe.qa_id = qp.id AND qe.model = :model
                WHERE df.dataset_id = :dsid
            )
            UNION ALL
            (
                SELECT
                    qp.id, qp.file_hash, qp.question_original, qp.question_norm,
                    qp.question_hash, qp.answer, qp.meta, qp.created_at,
                    qe.dim AS dim, qe.vec AS vec
                FROM qa_pairs qp
                JOIN qa_embeddings qe ON qe.qa_id = qp.id AND qe.model = :model
                WHERE qp.file_hash = :seed_fh
            )
            ORDER BY created_at DESC
            LIMIT :limit
            """
        )
        params = {"dsid": int(dataset_id), "model": model, "seed_fh": GLOBAL_SEED_FILE_HASH, "limit": int(limit)}
    else:
        sql = text(
            """
            SELECT
                qp.id, qp.file_hash, qp.question_original, qp.question_norm,
                qp.question_hash, qp.answer, qp.meta, qp.created_at,
                qe.dim AS dim, qe.vec AS vec
            FROM dataset_files df
            JOIN qa_pairs qp ON qp.file_hash = df.file_hash
            JOIN qa_embeddings qe ON qe.qa_id = qp.id AND qe.model = :model
            WHERE df.dataset_id = :dsid
            ORDER BY qp.created_at DESC
            LIMIT :limit
            """
        )
        params = {"dsid": int(dataset_id), "model": model, "limit": int(limit)}

    with eng.connect() as conn:
        rows = conn.execute(sql, params).mappings().all()

    result = []
    for row in rows:
        rec = _row_to_qarecord(row)
        result.append(
            (rec, int(row["dim"]) if row.get("dim") is not None else None, row.get("vec"))
        )
    return result

# New (lean projection): embedding-only candidates for a dataset
def repo_embedding_candidates_for_dataset(
    *,
    dataset_id: int,
    model: str,
    include_seeds: bool = True,
    limit: int = 200,
):
    """
    Return embedding candidates for semantic search as a lean projection.

    Columns returned (mappings):
      - qa_id (int)
      - question_norm (str)
      - dim (int)
      - vec (bytes)
      - src (str)  -> 'file' for dataset file QAs, 'seed' for global seeds

    Notes:
      - Uses only `qa_pairs.question_norm` + `qa_embeddings.vec` (no answers),
        because ranking works on questions-only.
      - Seeds are constrained to file_hash == GLOBAL_SEED_FILE_HASH AND qa_seed = 1.
    """
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")

    if include_seeds:
        sql = text(
            """
            (
                SELECT
                    qp.id        AS qa_id,
                    qp.question_norm,
                    qe.dim       AS dim,
                    qe.vec       AS vec,
                    'file'       AS src,
                    qp.created_at
                FROM dataset_files df
                JOIN qa_pairs qp
                  ON qp.file_hash = df.file_hash
                JOIN qa_embeddings qe
                  ON qe.qa_id = qp.id AND qe.model = :model
                WHERE df.dataset_id = :dsid
            )
            UNION ALL
            (
                SELECT
                    qp.id        AS qa_id,
                    qp.question_norm,
                    qe.dim       AS dim,
                    qe.vec       AS vec,
                    'seed'       AS src,
                    qp.created_at
                FROM qa_pairs qp
                JOIN qa_embeddings qe
                  ON qe.qa_id = qp.id AND qe.model = :model
                WHERE qp.file_hash = :seed_fh
                  AND COALESCE(qp.is_seed, 0) = 1
            )
            ORDER BY created_at DESC
            LIMIT :limit
            """
        )
        params = {
            "dsid": int(dataset_id),
            "model": model,
            "seed_fh": GLOBAL_SEED_FILE_HASH,
            "limit": int(limit),
        }
    else:
        sql = text(
            """
            SELECT
                qp.id        AS qa_id,
                qp.question_norm,
                qe.dim       AS dim,
                qe.vec       AS vec,
                'file'       AS src,
                qp.created_at
            FROM dataset_files df
            JOIN qa_pairs qp
              ON qp.file_hash = df.file_hash
            JOIN qa_embeddings qe
              ON qe.qa_id = qp.id AND qe.model = :model
            WHERE df.dataset_id = :dsid
            ORDER BY qp.created_at DESC
            LIMIT :limit
            """
        )
        params = {"dsid": int(dataset_id), "model": model, "limit": int(limit)}

    with eng.connect() as conn:
        rows = conn.execute(sql, params).mappings().all()

    # Strip created_at from the public return shape
    return [
        {
            "qa_id": int(r["qa_id"]),
            "question_norm": r["question_norm"] or "",
            "dim": int(r["dim"]) if r.get("dim") is not None else None,
            "vec": r["vec"],
            "src": r["src"],
        }
        for r in rows
    ]


# New (lean projection): embedding-only candidates for a file_hash
def repo_embedding_candidates_for_file(
    *,
    file_hash: str,
    model: str,
    include_seeds: bool = True,
    limit: int = 200,
):
    """
    Same as repo_embedding_candidates_for_dataset(), but restricted to a given file_hash.
    Returns a list of mappings with keys: qa_id, question_norm, dim, vec, src.
    """
    eng = get_engine()
    if eng is None:
        raise RuntimeError("DB engine not configured on current_app")

    if include_seeds:
        sql = text(
            """
            (
                SELECT
                    qp.id        AS qa_id,
                    qp.question_norm,
                    qe.dim       AS dim,
                    qe.vec       AS vec,
                    'file'       AS src,
                    qp.created_at
                FROM qa_pairs qp
                JOIN qa_embeddings qe
                  ON qe.qa_id = qp.id AND qe.model = :model
                WHERE qp.file_hash = :fh
            )
            UNION ALL
            (
                SELECT
                    qp.id        AS qa_id,
                    qp.question_norm,
                    qe.dim       AS dim,
                    qe.vec       AS vec,
                    'seed'       AS src,
                    qp.created_at
                FROM qa_pairs qp
                JOIN qa_embeddings qe
                  ON qe.qa_id = qp.id AND qe.model = :model
                WHERE qp.file_hash = :seed_fh
                  AND COALESCE(qp.is_seed, 0) = 1
            )
            ORDER BY created_at DESC
            LIMIT :limit
            """
        )
        params = {
            "fh": file_hash,
            "model": model,
            "seed_fh": GLOBAL_SEED_FILE_HASH,
            "limit": int(limit),
        }
    else:
        sql = text(
            """
            SELECT
                qp.id        AS qa_id,
                qp.question_norm,
                qe.dim       AS dim,
                qe.vec       AS vec,
                'file'       AS src,
                qp.created_at
            FROM qa_pairs qp
            JOIN qa_embeddings qe
              ON qe.qa_id = qp.id AND qe.model = :model
            WHERE qp.file_hash = :fh
            ORDER BY qp.created_at DESC
            LIMIT :limit
            """
        )
        params = {"fh": file_hash, "model": model, "limit": int(limit)}

    with eng.connect() as conn:
        rows = conn.execute(sql, params).mappings().all()

    return [
        {
            "qa_id": int(r["qa_id"]),
            "question_norm": r["question_norm"] or "",
            "dim": int(r["dim"]) if r.get("dim") is not None else None,
            "vec": r["vec"],
            "src": r["src"],
        }
        for r in rows
    ]