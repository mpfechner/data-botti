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
from helpers import sha256_bytesio, save_gzip_to_data, insert_dataset_and_file, analyze_and_store_columns

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


# Stelle sicher, dass der Upload-Ordner existiert
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# .env Variablen laden
load_dotenv()


# SQLAlchemy Engine erstellen
engine = create_engine(
    f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@db:3306/{os.getenv('MYSQL_DATABASE')}"
)


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
            user_id=1,                 # TODO: echten User verwenden, sobald vorhanden
            name=file.filename,
            file_info=file_info
        )

        # Weiter zur Analyse über dataset_id
        return redirect(url_for('analyze_dataset', dataset_id=dataset_id))

    return render_template('index.html')


@app.route('/analyze/dataset/<int:dataset_id>')
def analyze_dataset(dataset_id):
    # Metadaten zur archivierten Datei holen
    with engine.begin() as conn:
        row = conn.execute(
            text("""
                SELECT file_path, encoding, delimiter
                FROM dataset_files
                WHERE dataset_id = :id
            """),
            {"id": dataset_id}
        ).mappings().first()

    if not row:
        return f"Dataset {dataset_id} nicht gefunden.", 404

    file_path = row["file_path"]
    encoding = row["encoding"] or "utf-8"
    delimiter = row["delimiter"] or ","

    # CSV aus dem .gz lesen und analysieren
    with gzip.open(file_path, "rt", encoding=encoding, newline="") as f:
        df = pd.read_csv(f, delimiter=delimiter)

    # Spaltenanalyse berechnen und in DB schreiben
    analyze_and_store_columns(engine, dataset_id, df)

    summary = {
        "filename": os.path.basename(file_path),
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "head": df.head(10).to_html(classes="table table-striped", border=0),
        "description": df.describe(include="all").to_html(classes="table table-bordered", border=0),
    }
    # Spalteninfos aus der DB abrufen
    with engine.begin() as conn:
        columns = conn.execute(
            text("""
                 SELECT ordinal, name, dtype, is_nullable, distinct_count, min_val, max_val
                 FROM dataset_columns
                 WHERE dataset_id = :id
                 ORDER BY ordinal
                 """),
            {"id": dataset_id}
        ).mappings().all()

    return render_template("result.html", summary=summary, columns=columns)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
