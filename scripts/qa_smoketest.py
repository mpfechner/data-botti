# scripts/qa_smoketest.py
from app import app
from services.qa_service import normalize_question, hash_question, save_qa, find_exact_qa

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
            print("Found:", dict(row) if row else None)

        # Duplicate-Test
        try:
            save_qa(file_hash, q, qn, qh)
            print("Duplicate insert UNERWARTET erfolgreich")
        except Exception as e:
            print("Duplicate insert korrekt abgefangen:", type(e).__name__, str(e)[:120])

if __name__ == "__main__":
    main()