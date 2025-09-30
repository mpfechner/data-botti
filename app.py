# DataBotti – Webapp zur Analyse von CSV-Dateien
# Webapp zur Analyse von CSV-Dateien, mit visuellen Auswertungen und Berichtsexport


from flask import Flask, render_template, current_app, session
from flask_wtf.csrf import CSRFProtect
import os
from dotenv import load_dotenv
from db import init_engine
from routes.datasets import datasets_bp
from routes.assistant import assistant_bp
from routes.auth import auth_bp
from routes.admin import bp as admin_bp
from api import api_v1
from infra.auth_helpers import login_required, consent_required
from infra.config import get_config
from infra.logging import setup_app_logging

from sqlalchemy import text

# Load environment variables early so infra.config sees them
load_dotenv()

app = Flask(__name__)
csrf = CSRFProtect(app)
setup_app_logging(app)

# Blueprints an die App „andocken“
app.register_blueprint(datasets_bp)
app.register_blueprint(assistant_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(api_v1)
csrf.exempt(api_v1)

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


@app.route('/app', methods=['GET'])
@login_required
@consent_required
def app_dashboard():
    engine = current_app.config.get('DB_ENGINE')
    uid = session.get('user_id')
    groups = []
    if engine is not None and uid is not None:
        try:
            with engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT g.id, g.name
                        FROM groups g
                        JOIN user_groups ug ON ug.group_id = g.id
                        WHERE ug.user_id = :uid
                        ORDER BY g.name
                        """
                    ),
                    {"uid": uid},
                ).mappings().all()
                groups = [{"id": int(r["id"]), "name": r["name"]} for r in rows]
        except Exception:
            current_app.logger.exception("Failed to load user groups for /app")
    return render_template('app.html', groups=groups)


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
