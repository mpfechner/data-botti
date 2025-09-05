from flask import Blueprint, render_template, current_app, request, redirect, url_for
from sqlalchemy import text
import os
from helpers import load_csv_resilient, compute_generic_insights, analyze_and_store_columns
from services import storage

datasets_bp = Blueprint("datasets", __name__)

@datasets_bp.route("/upload", methods=["POST"])
def upload_dataset():
    file = request.files.get("file")
    if not file or file.filename == '':
        return render_template('index.html', error="Bitte eine CSV-Datei auswählen.")
    if not file.filename.lower().endswith('.csv'):
        return render_template('index.html', error="Nur .csv-Dateien sind erlaubt.")

    dataset_id = storage.save_uploaded_file(file, current_app.config["DB_ENGINE"])

    return redirect(url_for("datasets.analyze_dataset", dataset_id=dataset_id))


@datasets_bp.route("/analyze/dataset/<int:dataset_id>", endpoint="analyze_dataset")
def analyze_dataset(dataset_id):
    engine = current_app.config["DB_ENGINE"]

    # Metadaten zur archivierten Datei holen (inkl. original_name & file_hash)
    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT file_path, encoding, delimiter, original_name, file_hash
                FROM dataset_files
                WHERE dataset_id = :id
                """
            ),
            {"id": dataset_id},
        ).mappings().first()

    if not row:
        return f"Dataset {dataset_id} nicht gefunden.", 404

    file_path = row["file_path"]
    encoding = row["encoding"] or "utf-8"
    delimiter = row["delimiter"] or ","

    # CSV robust einlesen
    try:
        df, used_encoding, used_delimiter = load_csv_resilient(
            file_path,
            preferred_encoding=encoding,
            preferred_delimiter=delimiter,
        )
    except Exception as e:
        return f"Datei konnte nicht robust eingelesen werden: {e}", 400

    # Hinweise/Warnungen für den User
    warnings = []
    if hasattr(df, "attrs") and df.attrs.get("header_detected") is False:
        warnings.append(
            "Kein Header erkannt – Spalten wurden generisch benannt (col_0, col_1, …). Einige Auswertungen sind ggf. eingeschränkt."
        )
    if used_delimiter == ";":
        warnings.append("Semikolon als Trennzeichen erkannt (Excel-CSV).")

    # Generische Auto-Insights
    insights = compute_generic_insights(df)
    if isinstance(insights, dict) and insights.get("warnings"):
        warnings.extend(insights["warnings"])

    # Spaltenanalyse berechnen und in DB schreiben
    analyze_and_store_columns(engine, dataset_id, df)

    summary = {
        "filename": row["original_name"],
        "stored_filename": os.path.basename(file_path),
        "file_hash": row["file_hash"],
        "warnings": warnings,
        "shape": df.shape,
        "encoding_used": used_encoding,
        "delimiter_used": used_delimiter,
        "columns": df.columns.tolist(),
        "head": df.head(10).to_html(classes="table table-striped", border=0),
        "description": df.describe(include="all").to_html(classes="table table-bordered", border=0),
        "insights": insights,
    }

    with engine.begin() as conn:
        columns = conn.execute(
            text(
                """
                 SELECT ordinal, name, dtype, is_nullable, distinct_count, min_val, max_val
                 FROM dataset_columns
                 WHERE dataset_id = :id
                 ORDER BY ordinal
                 """
            ),
            {"id": dataset_id},
        ).mappings().all()

    ai_available = bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_BOTTI"))
    current_app.logger.debug("ai_available = %s", ai_available)
    current_app.logger.debug("Dataset %s: shape=%s", dataset_id, df.shape)
    current_app.logger.debug("Dataset %s: columns=%s", dataset_id, df.columns.tolist())
    current_app.logger.debug("Dataset %s: insights keys=%s", dataset_id, list(insights.keys()) if isinstance(insights, dict) else type(insights))
    return render_template(
        "result.html",
        summary=summary,
        columns=columns,
        dataset_id=dataset_id,
        ai_available=ai_available,
    )