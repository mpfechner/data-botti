import gzip
import hashlib
import io
import os
from sqlalchemy import text
from repo import insert_dataset_and_file

def save_uploaded_file(file, engine, user_id: int | None = None, group_ids: list[str] | None = None):
    buf = io.BytesIO(file.read())
    size_bytes = buf.getbuffer().nbytes
    hexhash = sha256_bytesio(buf)

    encoding = "utf-8"
    delimiter = ","

    os.makedirs("data", exist_ok=True)
    file_path = save_gzip_to_data(buf, hexhash, data_dir="data")

    # Normalize selected groups from request (if provided)
    sel_groups: list[int] = []
    if group_ids:
        try:
            sel_groups = [int(g) for g in group_ids if str(g).strip()]
        except Exception:
            raise ValueError("Ungültige Gruppen-Auswahl")

    # If groups are provided, we require an authenticated user_id
    if sel_groups and not user_id:
        raise ValueError("Gruppen können nur mit angemeldeten Benutzern gesetzt werden")

    # Create or get user (keep default helper if none provided)
    if not user_id:
        user_id = get_or_create_default_user(engine)

    # Insert dataset and its first file via repo-layer within a transaction
    with engine.begin() as conn:
        dataset_id, is_new = insert_dataset_and_file(
            conn,
            user_id=user_id,
            original_name=file.filename,
            file_hash=hexhash,
            file_path=file_path,
            size_bytes=size_bytes,
            encoding=encoding,
            delimiter=delimiter,
        )
        # If groups were selected, validate membership and link dataset to groups
        if sel_groups:
            # Validate membership: user must belong to every selected group
            for gid in set(sel_groups):
                row = conn.execute(
                    text("SELECT 1 FROM user_groups WHERE user_id = :uid AND group_id = :gid LIMIT 1"),
                    {"uid": int(user_id), "gid": int(gid)},
                ).first()
                if row is None:
                    raise ValueError("Ungültige Gruppe ausgewählt")
            # Create links (idempotent)
            for gid in set(sel_groups):
                conn.execute(
                    text(
                        """
                        INSERT INTO datasets_groups (dataset_id, group_id)
                        VALUES (:did, :gid)
                        ON DUPLICATE KEY UPDATE granted_at = granted_at
                        """
                    ),
                    {"did": int(dataset_id), "gid": int(gid)},
                )
    return dataset_id, is_new


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