from flask import Blueprint, render_template, request, current_app, session, redirect, url_for
from services.ai_client import ask_model, call_model
from services.ai_router import choose_model
from services.ai_tasks import build_relevant_columns_prompt
from services.ai_tasks import build_system_prompt, select_relevant_columns, build_final_prompt
from repo import get_latest_dataset_file
from helpers import (
    get_dataset_original_name,
    build_dataset_context,
    load_csv_resilient,
    summarize_columns_for_selection,
    build_cross_column_overview,
)
from datetime import datetime, timedelta, timezone
import pandas as _pd

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

        # Auto-infer expected_output for free prompts (no button/task) in a robust way
        try:
            eo_raw = (expected_output or "medium").strip().lower()
            # Only override when user likely typed a free prompt (no task button) and EO is default 'medium'
            if (not task) and eo_raw == "medium":
                txt = (prompt or "").lower()
                L = len(txt)
                # Length-based baseline
                if L < 120:
                    inferred = "short"
                elif L > 300:
                    inferred = "long"
                else:
                    inferred = "medium"

                # Keyword nudges (de/en)
                short_kw = ("kurz", "knapp", "bullet", "liste", "tl;dr")
                long_kw  = ("ausführlich", "detailliert", "begründe", "erkläre", "warum", "strategie", "management-zusammenfassung")
                if any(k in txt for k in long_kw):
                    inferred = "long"
                elif any(k in txt for k in short_kw):
                    inferred = "short"

                # Dataset size as corrective: many cols or very large rows -> bump one level up
                def _bump(eo: str) -> str:
                    return "medium" if eo == "short" else ("long" if eo == "medium" else "long")

                try:
                    _rows, _cols = int(rows), int(cols)
                except Exception:
                    _rows, _cols = 0, 0

                if _cols >= 20 or _rows >= 100_000:
                    inferred = _bump(inferred)

                expected_output = inferred
                current_app.logger.info(
                    "Auto-inferred expected_output=%s (len=%s, rows=%s, cols=%s)", inferred, L, _rows, _cols
                )
        except Exception:
            # Be conservative: keep original expected_output on any error
            pass

        # Domain-neutral system prompt: do not assume data are sensors, finance, etc.
        system_prompt = build_system_prompt()

        selection_prompt = build_relevant_columns_prompt(prompt, rows, cols, column_summaries)
        from services.ai_router import MODEL_FAST
        selection_result = ask_model(selection_prompt, expected_output="short", context_id=f"{dataset_id}_colsel", model=MODEL_FAST)

        selected_cols = select_relevant_columns(
            df,
            selection_result or "",
            task,
            rows,
            cols,
            column_summaries,
        )

        # Force-include: columns explicitly mentioned in the user's prompt text (preserve existing behavior)
        prompt_l = prompt.lower()
        for c in df.columns:
            if c.lower() in prompt_l and c not in selected_cols:
                selected_cols.append(c)

        current_app.logger.debug(
            "Stage-1 selected columns: %s | task=%s",
            selected_cols, task,
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

        # Build final Stage-2 prompt from blocks
        final_prompt = build_final_prompt(
            user_prompt=prompt,
            selected_cols=selected_cols,
            task=task,
            context=context,
            cross_overview=cross_overview,
            corr_block=corr_block,
        )

        # User override: allow forcing SMART model via checkbox
        force_smart = request.form.get("force_smart") == "1"
        if force_smart:
            from services.ai_router import MODEL_SMART
            model = MODEL_SMART
            current_app.logger.info(
                "AI model chosen (user-forced SMART): %s (expected_output=%s)", model, expected_output
            )
        else:
            model = choose_model(expected_output=expected_output, cache_ratio=None)
            current_app.logger.info(
                "AI model chosen: %s (expected_output=%s)", model, expected_output
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ]
        # Token budget: larger for GPT‑5 family to avoid empty outputs after reasoning
        if str(model).startswith("gpt-5"):
            _mt_map = {"short": 2000, "medium": 4000, "long": 8000}
        else:
            _mt_map = {"short": 400, "medium": 900, "long": 2000}
        _eo = (expected_output or "medium").lower()
        max_tokens = _mt_map.get(_eo, 900)
        current_app.logger.info("Token budget: model=%s eo=%s max_tokens=%s", model, _eo, max_tokens)

        result = call_model(model=model, messages=messages, max_tokens=max_tokens, temperature=0.2)
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