import gzip
import hashlib
import io
import os
from sqlalchemy import text
from repo import insert_dataset_and_file

def save_uploaded_file(file, engine):
    buf = io.BytesIO(file.read())
    size_bytes = buf.getbuffer().nbytes
    hexhash = sha256_bytesio(buf)

    encoding = "utf-8"
    delimiter = ","

    os.makedirs("data", exist_ok=True)
    file_path = save_gzip_to_data(buf, hexhash, data_dir="data")

    # Create or get default user (keeps using helper for now)
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