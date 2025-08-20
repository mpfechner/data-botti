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
from helpers import sha256_bytesio, save_gzip_to_data, insert_dataset_and_file

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
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('analyze', filename=file.filename))
    return render_template('index.html')


@app.route('/analyze/<filename>')
def analyze(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)

    summary = {
        'filename': filename,
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'head': df.head(10).to_html(classes='table table-striped', border=0),
        'description': df.describe(include='all').to_html(classes='table table-bordered', border=0)
    }
    return render_template('result.html', summary=summary)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
