import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import app as flask_app
from services.models import QARecord
from services.qa_service import (
    normalize_question,
    hash_question,
    save_qa,
    find_exact_qa,
    embed_question,
    save_embedding,
    find_semantic_candidates,
)

def main():
    with flask_app.app_context():
        file_hash = "demo-file-abc123"
        q = "Wie groß ist die Datei?"
        qn = normalize_question(q)
        qh = hash_question(qn)

        # 1) Insert (idempotent – nur anlegen, wenn noch nicht vorhanden)
        row = find_exact_qa(file_hash=file_hash, question_hash=qh)
        if row:
            qa_id = int(row.id) if isinstance(row, QARecord) else int(row["id"])  # compatible with QARecord or Mapping
            print(f"Reusing existing QA id: {qa_id}")
        else:
            qa_id = save_qa(
                file_hash=file_hash,
                question_original=q,
                question_norm=qn,
                question_hash=qh,
                answer=None,
                meta=None,
            )
            print(f"Inserted QA id: {qa_id}")

        # 2) Embedding berechnen + speichern
        vec = embed_question(qn)
        print(f"Embedding dim: {len(vec)} (first 5 dims: {vec[:5]})")
        save_embedding(qa_id=qa_id, vec=vec)
        print("Saved embedding.")

        # 3) Semantische Kandidaten für dieselbe Datei abfragen
        cands = find_semantic_candidates(file_hash=file_hash, query_vec=vec, k=5)
        print("Top candidates:", cands)

        # Erwartung: erster Eintrag ist (qa_id, score ~ 1.0)
        if cands and cands[0][0] == qa_id and cands[0][1] > 0.95:
            print("OK: self-similarity ≈ 1.0")
        else:
            print("WARN: expected self-similarity near 1.0")

if __name__ == "__main__":
    main()