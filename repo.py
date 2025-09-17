from __future__ import annotations
from typing import Optional, Mapping
from sqlalchemy import text
from flask import current_app
from sqlalchemy.exc import IntegrityError


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

def _get_engine_from_app():
    try:
        return current_app.config.get("DB_ENGINE")
    except Exception:
        return None


def repo_list_users():
    """Return all users with id, email, username, is_admin."""
    engine = _get_engine_from_app()
    if engine is None:
        raise RuntimeError("DB engine not configured on current_app")
    with engine.connect() as conn:
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
    engine = _get_engine_from_app()
    if engine is None:
        raise RuntimeError("DB engine not configured on current_app")
    with engine.begin() as conn:
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
    engine = _get_engine_from_app()
    if engine is None:
        raise RuntimeError("DB engine not configured on current_app")
    with engine.begin() as conn:
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
    engine = _get_engine_from_app()
    if engine is None:
        raise RuntimeError("DB engine not configured on current_app")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM users WHERE id = :user_id"), {"user_id": int(user_id)})


# --- group & membership repository helpers -----------------------------------------------------

def repo_list_groups():
    """Return all groups with member_count.
    Columns: id, name, created_at, member_count
    """
    engine = _get_engine_from_app()
    if engine is None:
        raise RuntimeError("DB engine not configured on current_app")
    with engine.connect() as conn:
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
    engine = _get_engine_from_app()
    if engine is None:
        raise RuntimeError("DB engine not configured on current_app")
    with engine.begin() as conn:
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
    engine = _get_engine_from_app()
    if engine is None:
        raise RuntimeError("DB engine not configured on current_app")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM groups WHERE id = :id"), {"id": int(group_id)})


def repo_list_group_members(*, group_id: int):
    """Return members of a group with user info.
    Columns: id, email, username, is_admin, added_at
    """
    engine = _get_engine_from_app()
    if engine is None:
        raise RuntimeError("DB engine not configured on current_app")
    with engine.connect() as conn:
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
    engine = _get_engine_from_app()
    if engine is None:
        raise RuntimeError("DB engine not configured on current_app")
    with engine.begin() as conn:
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
    engine = _get_engine_from_app()
    if engine is None:
        raise RuntimeError("DB engine not configured on current_app")
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM user_groups WHERE user_id = :uid AND group_id = :gid"),
            {"uid": int(user_id), "gid": int(group_id)},
        )

# --- admin guardrail helpers -------------------------------------------------------------------

def repo_is_user_admin(*, user_id: int) -> bool:
    engine = _get_engine_from_app()
    if engine is None:
        raise RuntimeError("DB engine not configured on current_app")
    with engine.connect() as conn:
        val = conn.execute(text("SELECT is_admin FROM users WHERE id = :id"), {"id": int(user_id)}).scalar()
    try:
        return bool(int(val)) if val is not None else False
    except Exception:
        return False


def repo_count_admins() -> int:
    engine = _get_engine_from_app()
    if engine is None:
        raise RuntimeError("DB engine not configured on current_app")
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM users WHERE is_admin = 1")).scalar()
    try:
        return int(n) if n is not None else 0
    except Exception:
        return 0

# --- qa_pairs repository helpers ---------------------------------------------------------------

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
    engine = _get_engine_from_app()
    if engine is None:
        raise RuntimeError("DB engine not configured on current_app")
    with engine.begin() as conn:
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
                "meta": meta,
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


def repo_qa_find_by_hash(*, file_hash: str, question_hash: str):
    """Find qa_pairs row by file_hash + question_hash. Returns mapping or None."""
    engine = _get_engine_from_app()
    if engine is None:
        raise RuntimeError("DB engine not configured on current_app")
    with engine.connect() as conn:
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
    return row