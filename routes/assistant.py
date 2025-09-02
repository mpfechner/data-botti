from flask import Blueprint, render_template, request, current_app, session, redirect, url_for
from services.ai_client import ask_model
from services.ai_tasks import build_chat_prompt
from helpers import get_dataset_original_name, build_dataset_context

assistant_bp = Blueprint("assistant", __name__)


@assistant_bp.route("/ai/<int:dataset_id>", methods=["GET", "POST"])
def ai_prompt(dataset_id):
    engine = current_app.config["DB_ENGINE"]
    filename = get_dataset_original_name(engine, dataset_id)

    ai_consent = bool(session.get("ai_consent", False))

    if request.method == "POST":
        if request.form.get("consent") == "1":
            session["ai_consent"] = True
            return redirect(url_for("assistant.ai_prompt", dataset_id=dataset_id))

        if not ai_consent:
            return render_template("ai_prompt.html", filename=filename, dataset_id=dataset_id, ai_consent=False)

        prompt = request.form.get("prompt", "")
        expected_output = request.form.get("expected_output", "medium")

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

        context = build_dataset_context(engine, dataset_id, n_rows=5, max_cols=12)
        final_prompt = f"{context}\n\n### Aufgabe\n{prompt}"

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