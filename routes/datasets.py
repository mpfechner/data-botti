from flask import Blueprint, render_template, current_app, request, redirect, url_for, flash
import os
from services import storage
from services.dataset_service import analyze_dataset as analyze_dataset_service
from services.ai_client import is_ai_available

datasets_bp = Blueprint("datasets", __name__)

@datasets_bp.route("/upload", methods=["POST"])
def upload_dataset():
    file = request.files.get("file")
    if not file or file.filename == '':
        return render_template('app.html', error="Bitte eine CSV-Datei auswählen.")
    if not file.filename.lower().endswith('.csv'):
        return render_template('app.html', error="Nur .csv-Dateien sind erlaubt.")

    dataset_id, is_new = storage.save_uploaded_file(file, current_app.config["DB_ENGINE"])
    if not is_new:
        flash("Diese Datei liegt bereits vor – vorhandenes Dataset wurde geöffnet.", "info")
    return redirect(url_for("datasets.analyze_dataset", dataset_id=dataset_id))


@datasets_bp.route("/analyze/dataset/<int:dataset_id>", endpoint="analyze_dataset")
def analyze_dataset(dataset_id):
    engine = current_app.config["DB_ENGINE"]

    summary, columns = analyze_dataset_service(engine, dataset_id)

    ai_available = is_ai_available()
    current_app.logger.debug("ai_available = %s", ai_available)
    try:
        current_app.logger.debug("Dataset %s: shape=%s", dataset_id, summary.get("shape"))
        current_app.logger.debug("Dataset %s: columns=%s", dataset_id, summary.get("columns"))
        insights = summary.get("insights")
        current_app.logger.debug(
            "Dataset %s: insights keys=%s",
            dataset_id,
            list(insights.keys()) if isinstance(insights, dict) else type(insights),
        )
    except Exception:
        pass

    return render_template(
        "result.html",
        summary=summary,
        columns=columns,
        dataset_id=dataset_id,
        ai_available=ai_available,
    )