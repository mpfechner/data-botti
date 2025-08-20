import hashlib, gzip, io, os
from sqlalchemy import text

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