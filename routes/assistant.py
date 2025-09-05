from flask import Blueprint, render_template, request, current_app, session, redirect, url_for
from services.ai_client import ask_model
from services.ai_tasks import build_relevant_columns_prompt
from sqlalchemy import text as _sql_text
from helpers import (
    get_dataset_original_name,
    build_dataset_context,
    load_csv_resilient,
    summarize_columns_for_selection,
    build_cross_column_overview,
)

assistant_bp = Blueprint("assistant", __name__)


@assistant_bp.route("/ai/<int:dataset_id>", methods=["GET", "POST"])
def ai_prompt(dataset_id):
    engine = current_app.config["DB_ENGINE"]
    filename = get_dataset_original_name(engine, dataset_id)

    ai_consent = bool(session.get("ai_consent", False))

    # Load dataset from latest stored file_path (gz CSV)
    with engine.begin() as conn:
        meta = conn.execute(
            _sql_text(
                """
                SELECT file_path, encoding, delimiter
                FROM dataset_files
                WHERE dataset_id = :id
                ORDER BY id DESC
                LIMIT 1
                """
            ),
            {"id": int(dataset_id)},
        ).mappings().first()

    if not meta:
        current_app.logger.error("No dataset_files meta for dataset_id=%s", dataset_id)
        return render_template(
            "ai_prompt.html",
            filename=filename,
            dataset_id=dataset_id,
            ai_consent=ai_consent,
            error="Kein Datei-Metadatensatz gefunden. Bitte erneut hochladen.",
        )

    file_path = meta["file_path"]
    preferred_enc = meta.get("encoding") if isinstance(meta, dict) else meta["encoding"]
    preferred_delim = meta.get("delimiter") if isinstance(meta, dict) else meta["delimiter"]

    try:
        df, used_enc, used_delim = load_csv_resilient(
            file_path=file_path,
            preferred_encoding=preferred_enc,
            preferred_delimiter=preferred_delim,
        )
    except Exception as e:
        current_app.logger.exception("Failed to load CSV for dataset_id=%s: %s", dataset_id, e)
        return render_template(
            "ai_prompt.html",
            filename=filename,
            dataset_id=dataset_id,
            ai_consent=ai_consent,
            error=f"CSV konnte nicht geladen werden: {e}",
        )

    rows, cols = df.shape
    column_summaries = summarize_columns_for_selection(df)

    if request.method == "POST":
        if request.form.get("consent") == "1":
            session["ai_consent"] = True
            return redirect(url_for("assistant.ai_prompt", dataset_id=dataset_id))

        if not ai_consent:
            return render_template("ai_prompt.html", filename=filename, dataset_id=dataset_id, ai_consent=False)

        prompt = request.form.get("prompt", "")
        expected_output = request.form.get("expected_output", "medium")
        task = request.form.get("task", "").strip().lower()
        current_app.logger.debug("Task received: %s", task)

        # Strenger System-Prompt
        system_prompt = (
            "Du bist DataBotti, ein Assistent für die Analyse von Sensordaten (CSV). "
            "Antworte ausschließlich anhand des bereitgestellten Datensatz-Kontexts (Spalten, Beispielzeilen, Statistiken). "
            "Wenn der Kontext fehlt, sage das klar und liste knapp auf, was du brauchst. "
            "Erfinde keine Geschäftskennzahlen oder externen Fakten. "
            "Benenne Unsicherheit ausdrücklich, wenn Evidenz im Kontext fehlt oder unklar ist (z.B. wenn keine Extremwerte oder Verteilungsdaten vorliegen). "
            "Stütze Aussagen auf konkrete Zahlen aus dem Kontext (z.B. Min/Max, Quantile, Outlier-Anteile) und markiere Hypothesen als solche. "
            "Gib keine sensiblen oder externen Daten aus und lehne Spekulationen ohne Datengrundlage ab."
        )

        selection_prompt = build_relevant_columns_prompt(prompt, rows, cols, column_summaries)
        selection_result = ask_model(selection_prompt, expected_output="short", context_id=f"{dataset_id}_colsel")
        # Parse result
        raw_selected: list[str] = []
        if selection_result:
            txt = selection_result.strip()
            if txt.upper() == "ALL":
                raw_selected = list(df.columns)
            else:
                raw_selected = [c.strip() for c in txt.split(",") if c.strip()]

        # Harden: allow only existing cols, drop dups, preserve order
        seen = set()
        selected_cols = []
        for c in raw_selected:
            if c in df.columns and c not in seen:
                seen.add(c)
                selected_cols.append(c)

        # Force-include: columns explicitly mentioned in the user's prompt text
        prompt_l = prompt.lower()
        force_include = [c for c in df.columns if c.lower() in prompt_l]
        for c in force_include:
            if c not in seen:
                seen.add(c)
                selected_cols.append(c)

        # Task-based overrides / expansions
        import pandas as _pd
        num_cols = list(df.select_dtypes(include=["number"]).columns)
        cat_cols = list(df.select_dtypes(include=["object", "string", "category"]).columns)
        # datetime heuristic: try parse a small sample
        dt_cols: list[str] = []
        for c in df.columns:
            try:
                s = _pd.to_datetime(df[c], errors="coerce", utc=False)
                if s.notna().mean() > 0.5:
                    dt_cols.append(c)
            except Exception:
                continue

        def _ensure(cols: list[str]):
            for c in cols:
                if c in df.columns and c not in seen:
                    seen.add(c)
                    selected_cols.append(c)

        if task in {"summary", "quality", "correlations"}:
            # Für generische Aufgaben: mit ALL arbeiten
            selected_cols = list(df.columns)
            seen = set(selected_cols)
        elif task == "outliers":
            # Numerische Spalten sind relevant
            _ensure(num_cols)
        elif task == "categories":
            # Kategoriale Spalten sicherstellen
            _ensure(cat_cols)
        elif task == "time_trends":
            # Datums- + numerische Spalten
            _ensure(dt_cols + num_cols)
        elif task == "duplicates":
            # Für Eindeutigkeit/Duplikate: alle Spalten prüfen
            selected_cols = list(df.columns)
            seen = set(selected_cols)
        elif task == "segments":
            # Segmentvergleich: mindestens eine Kategorie + numerische
            if not any(c in selected_cols for c in cat_cols):
                _ensure(cat_cols[:1])
            _ensure(num_cols)

        # Fallback, falls leer
        if not selected_cols:
            selected_cols = list(df.columns)
            seen = set(selected_cols)

        current_app.logger.debug(
            "Stage-1 selected columns: %s | force_include=%s | task=%s",
            selected_cols, force_include, task,
        )

        # Cross-column overview always over FULL dataset for global context
        cross_overview = build_cross_column_overview(df)

        corr_block = ""
        if task == "correlations":
            # Always compute over ALL numeric columns for a complete view
            df_num = df.select_dtypes(include=["number"])
            if df_num.shape[1] >= 2:
                try:
                    corr = df_num.corr(method="pearson")
                    cols_num = list(df_num.columns)
                    pairs = []
                    for i in range(len(cols_num)):
                        for j in range(i + 1, len(cols_num)):
                            v = corr.iat[i, j]
                            if _pd.notna(v):
                                pairs.append((float(v), cols_num[i], cols_num[j]))
                    pos = sorted([p for p in pairs if p[0] > 0], key=lambda x: x[0], reverse=True)[:5]
                    neg = sorted([p for p in pairs if p[0] < 0], key=lambda x: x[0])[:5]

                    def _table(items, title):
                        if not items:
                            return ""
                        rows = "\n".join(f"| {a} | {b} | {v:.3f} |" for v, a, b in items)
                        return f"{title}\n\n| Spalte A | Spalte B | r |\n|---|---|---|\n{rows}\n"

                    corr_block = "### Korrelationen (vorberechnet, Pearson)\n\n"
                    corr_block += _table(pos, "Top 5 positiv:") or "_Keine positiven Korrelationen gefunden._\n"
                    corr_block += "\n"
                    corr_block += _table(neg, "Top 5 negativ:") or "_Keine negativen Korrelationen gefunden._\n"
                    current_app.logger.debug("Correlations computed: num_cols=%s pairs=%s pos=%s neg=%s", df_num.shape[1], len(pairs), len(pos), len(neg))
                except Exception as e:
                    current_app.logger.exception("Correlation computation failed: %s", e)
                    corr_block = "### Korrelationen (vorberechnet)\n(Berechnung fehlgeschlagen.)\n"
            else:
                corr_block = "### Korrelationen (vorberechnet)\n(Nicht genug numerische Spalten für Korrelationen.)\n"

        # Build context for Stage-2 with the selected columns
        context = build_dataset_context(engine, dataset_id, n_rows=5, include_columns=selected_cols)

        # Enforce coverage: if ALL columns were selected, say so explicitly; otherwise require covering all selected columns
        all_selected = len(selected_cols) == len(df.columns)
        if all_selected:
            coverage_note = (
                "Wichtiger Hinweis: Die Spaltenauswahl (Stufe 1) hat ALLE Spalten als relevant markiert. "
                "Analysiere daher den vollständigen Spaltensatz. Beziehe deine Aussagen auf alle Spalten. "
                "Wenn du gruppierst oder bündelst, kennzeichne das ausdrücklich."
            )
        else:
            coverage_note = (
                "Wichtiger Hinweis: Beziehe deine Analyse auf ALLE ausgewählten Spalten (nicht nur eine Teilmenge). "
                "Wenn du zur Übersicht gruppierst oder bündelst, erwähne das ausdrücklich."
            )

        final_prompt = (
            f"{context}\n\n"
            f"### Cross-Column Overview\n{cross_overview}\n\n"
            f"{corr_block}\n\n"
            f"### Aufgabe\n{prompt}\n\n"
            f"### Pflichtenheft\n{coverage_note}"
        )
        current_app.logger.debug("Stage-2 columns used: %s of %s (all_selected=%s)", len(selected_cols), len(df.columns), all_selected)

        result = ask_model(
            final_prompt,
            expected_output=expected_output,
            context_id=dataset_id,
            system_prompt=system_prompt,
        )
        return render_template(
            "ai_result.html",
            result=result,
            filename=filename,
            dataset_id=dataset_id,
            prompt=prompt,
        )

    return render_template("ai_prompt.html", filename=filename, dataset_id=dataset_id, ai_consent=ai_consent)