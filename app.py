# DataBotti – Webapp zur Analyse von CSV-Dateien
# Webapp zur Analyse von CSV-Dateien, mit visuellen Auswertungen und Berichtsexport


from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import hashlib
import gzip
import io
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import matplotlib.pyplot as plt
from helpers import sha256_bytesio, save_gzip_to_data, insert_dataset_and_file, analyze_and_store_columns, load_csv_resilient, get_or_create_default_user, compute_generic_insights, get_dataset_original_name, build_dataset_context
import logging
from logging.handlers import RotatingFileHandler
from routes.datasets import datasets_bp
from routes.assistant import assistant_bp
from infra.config import get_config
from infra.logging import get_logger

# Load environment variables early so infra.config sees them
load_dotenv()

app = Flask(__name__)

# Blueprints an die App „andocken“
app.register_blueprint(datasets_bp)
app.register_blueprint(assistant_bp)

# Apply config
app.config.update(get_config())

# Setup logger
logger = get_logger(__name__)
app.logger.handlers = logger.handlers
app.logger.setLevel(logger.level)
logger.info("Starting DataBotti application")


# Stelle sicher, dass der Upload-Ordner existiert
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# SQLAlchemy Engine erstellen
user = os.getenv("MYSQL_USER")
pwd  = os.getenv("MYSQL_PASSWORD")
db   = os.getenv("MYSQL_DATABASE")
host = os.getenv("MYSQL_HOST", "127.0.0.1")
port = os.getenv("MYSQL_PORT", "3306")

engine = create_engine(
    f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}",
    pool_pre_ping=True,         # prüft Connection vor Nutzung → verhindert 2013-Fehler
)
app.config['DB_ENGINE'] = engine


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('index.html', error="Bitte eine CSV-Datei auswählen.")
        if not file.filename.lower().endswith('.csv'):
            return render_template('index.html', error="Nur .csv-Dateien sind erlaubt.")

        # CSV in Memory lesen
        buf = io.BytesIO(file.read())
        size_bytes = buf.getbuffer().nbytes

        # Hash berechnen
        hexhash = sha256_bytesio(buf)

        # Vorerst feste Defaults (Auto-Detect machen wir später)
        encoding = "utf-8"
        delimiter = ","

        # Archiv speichern: data/<hash>.csv.gz
        os.makedirs("data", exist_ok=True)
        file_path = save_gzip_to_data(buf, hexhash, data_dir="data")

        # DB: datasets + dataset_files
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
            user_id=get_or_create_default_user(engine),  # TODO: echten User verwenden
            filename=file.filename,  # <-- vorher: name=...
            file_info=file_info
        )

        # Weiter zur Analyse über dataset_id
        return redirect(url_for('datasets.analyze_dataset', dataset_id=dataset_id))

    return render_template('index.html')


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
