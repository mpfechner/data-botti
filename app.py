# DataBotti – Webapp zur Analyse von CSV-Dateien
# Webapp zur Analyse von CSV-Dateien, mit visuellen Auswertungen und Berichtsexport


from flask import Flask, render_template
import os
from dotenv import load_dotenv
from db import init_engine
from routes.datasets import datasets_bp
from routes.assistant import assistant_bp
from infra.config import get_config
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


app.config['DB_ENGINE'] = init_engine()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
