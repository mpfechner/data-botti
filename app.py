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
from infra.logging import setup_app_logging

# Load environment variables early so infra.config sees them
load_dotenv()

app = Flask(__name__)
setup_app_logging(app)

# Blueprints an die App „andocken“
app.register_blueprint(datasets_bp)
app.register_blueprint(assistant_bp)

# Apply config
app.config.update(get_config())

# Setup logger
app.logger.info("Starting DataBotti application")


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


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
