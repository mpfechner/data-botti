# scripts/test_qa_smoketest.py
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.dirname(_HERE)
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)
try:
    from app import app as app
except Exception:
    from app import create_app  # type: ignore
    app = create_app()

from services.qa_service import normalize_question, hash_question, save_qa, find_exact_qa
from services.models import QARecord

def main():
    # App-Context aktivieren
    with app.app_context():
        file_hash = "demo-file-abc123"
        q = "Wie groß   ist  die Datei ? "

        # Normalisierung & Hash
        qn = normalize_question(q)
        qh = hash_question(qn)
        print("Normalized:", qn)
        print("Hash:", qh)

        # Insert
        try:
            qa_id = save_qa(
                file_hash=file_hash,
                question_original=q,
                question_norm=qn,
                question_hash=qh,
                answer=None,
                meta=None
            )
            print("Inserted qa_id:", qa_id)
        except Exception as e:
            print("Insert error:", type(e).__name__, str(e)[:120])
            qa_id = None

        # Exact-Match Lookup
        if qa_id:
            row = find_exact_qa(file_hash=file_hash, question_hash=qh)
            if row:
                if isinstance(row, QARecord):
                    print("Found:", {"id": row.id, "file_hash": row.file_hash, "question": row.question})
                else:
                    print("Found:", dict(row))
            else:
                print("Found:", None)

        # Duplicate-Test
        try:
            save_qa(file_hash, q, qn, qh)
            print("Duplicate insert UNERWARTET erfolgreich")
        except Exception as e:
            print("Duplicate insert korrekt abgefangen:", type(e).__name__, str(e)[:120])

if __name__ == "__main__":
    main()