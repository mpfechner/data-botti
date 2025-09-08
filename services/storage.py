import io
import os
from helpers import sha256_bytesio, save_gzip_to_data, get_or_create_default_user
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