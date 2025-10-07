import os
import sys
import hashlib
from pathlib import Path
from typing import List, Tuple
from sqlalchemy import create_engine, text

# --- Ensure project root is importable so "services.*" works when run as: python scripts/seed_qa_pairs.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Import embedding backfill from app code (re-uses the normal embedding pipeline)
from services.qa_service import backfill_missing_embeddings

# =====================================================================================
# Database connection setup
# Prefer DATABASE_URL if present; otherwise compose from MYSQL_* envs
# =====================================================================================
def _build_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        DB_USER = os.getenv("MYSQL_USER", "michael")
        DB_PASS = os.getenv("MYSQL_PASSWORD", "test123")
        DB_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
        DB_NAME = os.getenv("MYSQL_DATABASE", "databotti")
        DB_PORT = os.getenv("MYSQL_PORT", "3306")
        db_url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
    return create_engine(db_url)

engine = _build_engine()

# =====================================================================================
# Seed QA pairs
# Strategy:
# - Start from a compact base set of CSV-related Q/A.
# - Generate paraphrase variants to reach >=200 entries.
# - All seeds are stored under file_hash="seed_data" with is_seed=1 to keep them
#   isolated from user data and to avoid leaking into UI answers unless explicitly used.
# =====================================================================================

BASE_QA: List[Tuple[str, str]] = [
    ("Was ist eine CSV-Datei?",
     "Eine CSV-Datei (Comma Separated Values) speichert tabellarische Daten in Textform; jede Zeile ist ein Datensatz, Spalten sind durch ein Trennzeichen (z. B. Komma, Semikolon, Tab, Pipe) getrennt."),
    ("Wie erkenne ich den Delimiter einer CSV?",
     "Typische Delimiter sind Komma, Semikolon, Tab oder Pipe. Man kann ihn an der Kopfzeile erkennen oder automatisch ermitteln (z. B. mit Python csv.Sniffer())."),
    ("Welche Encodings kommen bei CSV häufig vor?",
     "Häufige Encodings sind UTF‑8 (empfohlen) und Latin‑1/ISO‑8859‑1. Bei Problemen hilft automatisches Fallback oder explizites Setzen des Encodings."),
    ("Wie gehe ich mit Dezimalkomma in CSV um?",
     "Werte mit Dezimalkomma (z. B. „1,23“) sollten in Dezimalpunkt („1.23“) konvertiert werden, bevor man sie numerisch auswertet."),
    ("Wie erkenne ich, ob die erste Zeile Header ist?",
     "Vergleiche die ersten Zeilen: Wenn Zeile 1 eher Bezeichner enthält und sich das Typmuster von Zeile 2 unterscheidet, ist Zeile 1 vermutlich der Header."),
    ("Wie lade ich eine gz‑komprimierte CSV robust?",
     "Datei mit gzip öffnen und mit Fallbacks für Encoding und Delimiter lesen; bei Fehlern alternative Delimiter/Encodings testen."),
    ("Wie gehe ich mit fehlerhaften Zeilen um?",
     "Beim Einlesen on_bad_lines='warn' (oder 'skip') verwenden, problematische Zeilen protokollieren und später gezielt bereinigen."),
    ("Wie finde/entferne ich Dubletten?",
     "Exakte Zeilenduplikate via vollständigem Zeilenvergleich oder Duplikate nach Schlüsselspalten (z. B. date, id) identifizieren und entfernen."),
    ("Wie erkenne ich fehlende Werte?",
     "Typische NA-Tokens (NA, NaN, null, leere Strings) als fehlend interpretieren und die Quote je Spalte berichten."),
    ("Wie prüfe ich Ausreißer in numerischen Spalten?",
     "IQR- oder z‑score‑basiert identifizieren; bei sehr kleinen Stichproben sind Ausreißerbewertung und Korrelationen unzuverlässig."),
    ("Wie parse ich Datums-Spalten?",
     "Einheitliches ISO‑Format bevorzugen und beim Laden Datum zu datetime konvertieren; Zeitzonen/Locale konsistent halten."),
    ("Wie erkenne ich Spaltennamen?",
     "Spaltennamen stehen im Header; wenn keiner vorhanden ist, generische Namen (col_0, col_1, …) vergeben und optional umbenennen."),
    ("Wie gehe ich mit unterschiedlich formatierten Städtenamen um?",
     "Werte normalisieren (Trim, einheitliche Groß-/Kleinschreibung, Umlaute), um Inkonsistenzen und Dubletten zu vermeiden."),
    ("Wie berechne ich Korrelationen?",
     "Mindestens zwei numerische Spalten und ausreichend Zeilen nötig; Pearson für lineare, Spearman für monotone Zusammenhänge."),
    ("Welche Mindeststichprobe ist sinnvoll?",
     "Kommt auf Zielgröße an. Für robuste Schätzungen meist ≥30 Beobachtungen; für Tests ggf. Power‑Berechnung durchführen."),
    ("Wie kann ich Spalten-Typen automatisch erkennen?",
      "Über Heuristiken: numerische Muster, Datums-/Zeitmuster, ansonsten Text. Danach typgerecht verarbeiten."),
    ("Wie protokolliere ich Einleseparameter?",
     "Encoding, Delimiter, Header-Entscheidung, NA-Tokens und erkannte Typen speichern, damit Ergebnisse reproduzierbar sind."),
    ("Wie gehe ich mit sehr breiten CSVs (viele Spalten) um?",
     "Eine Vorschau (n Zeilen × m Spalten) laden, Spaltenprofile berechnen und nur relevante Spalten in Analysen ziehen."),
    ("Welche typischen Fehler verursachen \"Encoding Error\"?",
     "Falsche Annahme über Encoding, gemischte Encodings in einer Datei oder Binärartefakte; Fallbacks und explizites Setzen helfen."),
    ("Wie messe ich Datenqualität kurz & knapp?",
     "Checkliste: Fehlwerte‑Quote, Dubletten, Ausreißer, Typkonsistenz, Kategorienormalisierung, Datums-/Zeitvalidität.")
]

# Paraphrase templates to create variants
PHRASINGS = [
    "Erkläre kurz: {}",
    "Kurz erklärt: {}",
    "Wie geht man vor: {}",
    "Best Practice: {}",
    "FAQ: {}",
    "Knapp beantworten: {}",
    "Praxis-Tipp: {}",
    "In einem Satz: {}",
]

def build_seed_pairs(target: int = 200) -> List[Tuple[str, str]]:
    # Start with base and then add paraphrases until target reached
    pairs = BASE_QA.copy()
    # Convert to list of tuples (question, answer)
    pairs = [(q, a) for (q, a) in pairs]
    qi = 0
    while len(pairs) < target:
        q0, a0 = BASE_QA[qi % len(BASE_QA)]
        tpl = PHRASINGS[(qi // len(BASE_QA)) % len(PHRASINGS)]
        q_new = tpl.format(q0)
        # Keep answer identical to maintain correctness
        pairs.append((q_new, a0))
        qi += 1
    return pairs[:target]

def upsert_seed_pairs(pairs: List[Tuple[str, str]], file_hash: str = "seed_data") -> tuple[int, int]:
    inserted, skipped = 0, 0
    with engine.begin() as conn:
        for q, a in pairs:
            q = q.strip()
            a = a.strip()
            q_hash = hashlib.sha256(q.encode("utf-8")).hexdigest()

            # Rely on unique(uq_qa_question_hash_file) to avoid duplicates per file_hash
            # Check fast path to avoid exception spam
            exists_sql = text("""
                SELECT COUNT(*) FROM qa_pairs
                WHERE question_hash = :qh AND file_hash = :fh
            """)
            exists = conn.execute(exists_sql, {"qh": q_hash, "fh": file_hash}).scalar()
            if exists:
                skipped += 1
                continue

            ins_sql = text("""
                INSERT INTO qa_pairs
                  (file_hash, question_original, question_norm, question_hash, answer, meta, is_seed)
                VALUES
                  (:fh, :qo, :qn, :qh, :a, JSON_OBJECT('source','seed'), 1)
            """)
            conn.execute(ins_sql, {
                "fh": file_hash,
                "qo": q,
                "qn": q.lower(),
                "qh": q_hash,
                "a": a
            })
            inserted += 1
    return inserted, skipped

def main():
    target = int(os.getenv("SEED_QA_TARGET", "200"))
    file_hash = os.getenv("SEED_FILE_HASH", "seed_data")

    pairs = build_seed_pairs(target=target)
    inserted, skipped = upsert_seed_pairs(pairs, file_hash=file_hash)
    print(f"[seed] qa_pairs: inserted={inserted}, skipped={skipped}, target={target}, file_hash={file_hash}")

    # Now create embeddings for any new pairs using the app's standard pipeline
    backfill_missing_embeddings(file_hash=file_hash)  # uses default model "intfloat/multilingual-e5-base"
    print("[seed] embeddings backfilled for new seed QA pairs.")

if __name__ == "__main__":
    main()
