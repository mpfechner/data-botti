import os
from helpers import load_csv_resilient, compute_generic_insights, analyze_and_store_columns
from repo import get_latest_dataset_file, get_dataset_columns


def analyze_dataset(engine, dataset_id):

    # Metadaten zur archivierten Datei holen (inkl. original_name & file_hash)
    with engine.begin() as conn:
        row = get_latest_dataset_file(conn, dataset_id)

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
        columns = get_dataset_columns(conn, dataset_id)

    return summary, columns