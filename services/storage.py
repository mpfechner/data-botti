

import io
import os
from helpers import sha256_bytesio, save_gzip_to_data, insert_dataset_and_file, get_or_create_default_user

def save_uploaded_file(file, engine):
    buf = io.BytesIO(file.read())
    size_bytes = buf.getbuffer().nbytes
    hexhash = sha256_bytesio(buf)

    encoding = "utf-8"
    delimiter = ","

    os.makedirs("data", exist_ok=True)
    file_path = save_gzip_to_data(buf, hexhash, data_dir="data")

    file_info = {
        "original_name": file.filename,
        "size_bytes": size_bytes,
        "file_hash": hexhash,
        "encoding": encoding,
        "delimiter": delimiter,
        "file_path": file_path
    }

    dataset_id = insert_dataset_and_file(
        engine,
        user_id=get_or_create_default_user(engine),
        filename=file.filename,
        file_info=file_info
    )
    return dataset_id