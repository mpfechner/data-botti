from flask import Blueprint, render_template, request, current_app, session, redirect, url_for
from services.ai_client import ask_model
from services.ai_tasks import build_relevant_columns_prompt
from repo import get_latest_dataset_file
from helpers import (
    get_dataset_original_name,
    build_dataset_context,
    load_csv_resilient,
    summarize_columns_for_selection,
    build_cross_column_overview,
)
from datetime import datetime, timedelta, timezone

assistant_bp = Blueprint("assistant", __name__)


def _now_utc():
    return datetime.now(timezone.utc)

def _get_consent_policy():
    """Read consent policy from config with safe defaults.
    CONSENT_VERSION: int (default 1)
    CONSENT_MAX_AGE_DAYS: int or None (default 7)
    """
    cfg = current_app.config
    version = int(cfg.get("CONSENT_VERSION", 1))
    max_age_days = cfg.get("CONSENT_MAX_AGE_DAYS", 7)
    try:
        max_age_days = int(max_age_days) if max_age_days is not None else None
    except Exception:
        max_age_days = 7
    return version, max_age_days

def _has_valid_consent() -> bool:
    """Server-side check: version + (optional) TTL."""
    if not session.get("ai_consent"):
        return False
    try:
        stored_version = int(session.get("consent_version", 0))
    except Exception:
        stored_version = 0
    consent_time_raw = session.get("consent_time")
    try:
        consent_time = (
            datetime.fromisoformat(consent_time_raw)
            if isinstance(consent_time_raw, str)
            else consent_time_raw
        )
    except Exception:
        consent_time = None

    policy_version, max_age_days = _get_consent_policy()
    if stored_version != policy_version:
        return False
    if max_age_days is not None:
        if consent_time is None:
            return False
        if _now_utc() - consent_time > timedelta(days=max_age_days):
            return False
    return True

def _grant_consent():
    policy_version, _ = _get_consent_policy()
    session["ai_consent"] = True
    session["consent_version"] = int(policy_version)
    session["consent_time"] = _now_utc().isoformat()

def _revoke_consent():
    for k in ("ai_consent", "consent_version", "consent_time"):
        session.pop(k, None)


@assistant_bp.route("/ai/<int:dataset_id>", methods=["GET", "POST"])
def ai_prompt(dataset_id):
    engine = current_app.config["DB_ENGINE"]
    filename = get_dataset_original_name(engine, dataset_id)

    # Strict, server-side consent validation (version + optional TTL)
    ai_consent = _has_valid_consent()

    # Handle POSTed consent action early
    if request.method == "POST" and request.form.get("consent") == "1":
        _grant_consent()
        return redirect(url_for("assistant.ai_prompt", dataset_id=dataset_id))

    # Gate both GET and POST if consent is not valid (no heavy work before consent)
    if not ai_consent and request.method == "GET":
        return render_template("ai_prompt.html", filename=filename, dataset_id=dataset_id, ai_consent=False)

    # Load dataset from latest stored file_path (gz CSV)
    with engine.begin() as conn:
        meta = get_latest_dataset_file(conn, dataset_id)

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
        # Defense-in-depth: block POST if consent is not valid
        if not _has_valid_consent():
            return render_template("ai_prompt.html", filename=filename, dataset_id=dataset_id, ai_consent=False)

        prompt = request.form.get("prompt", "")
        expected_output = request.form.get("expected_output", "medium")
        task = request.form.get("task", "").strip().lower()
        current_app.logger.debug("Task received: %s", task)

        # Domain-neutral system prompt: do not assume data are sensors, finance, etc.
        system_prompt = (
            "Du bist DataBotti, ein Assistent für die Analyse von tabellarischen CSV-Daten. "
            "Triff KEINE Annahmen über die Art der Daten (z. B. Sensordaten, Finanzdaten, Logdaten), "
            "es sei denn, dies geht explizit und eindeutig aus dem bereitgestellten Kontext hervor. "
            "Antworte ausschließlich anhand des Datensatz-Kontexts (Spalten, Beispielzeilen, Statistiken). "
            "Wenn der Kontext fehlt, sage das klar und liste knapp auf, was du brauchst. "
            "Erfinde keine Geschäftskennzahlen oder externen Fakten. "
            "Benenne Unsicherheit ausdrücklich, wenn Evidenz im Kontext fehlt oder unklar ist (z. B. wenn keine Extremwerte oder Verteilungsdaten vorliegen). "
            "Stütze Aussagen auf konkrete Zahlen aus dem Kontext (z. B. Min/Max, Quantile, Ausreißer-Anteile) und markiere Hypothesen als solche. "
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


@assistant_bp.route("/ai/revoke", methods=["POST"])
def revoke_ai_consent():
    _revoke_consent()
    ds_id = request.args.get("dataset_id", type=int)
    if ds_id:
        return redirect(url_for("assistant.ai_prompt", dataset_id=ds_id))
    return redirect(url_for("datasets.index"))