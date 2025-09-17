

# scripts/embedding_checks.py
import os
import sys

# Ensure project root on sys.path (robust when run from anywhere)
HERE = os.path.dirname(__file__)
ROOT = HERE  # file is at project root; adjust if you move it
PARENT = os.path.dirname(HERE)
for p in {HERE, ROOT, PARENT}:
    if p and p not in sys.path:
        sys.path.append(p)

try:
    from app import app as flask_app
except Exception:
    # Fallback: try plain import
    from app import app as flask_app

import numpy as np
from services.qa_service import (
    normalize_question,
    hash_question,
    save_qa,
    find_exact_qa,
    embed_question,
    save_embedding,
    find_semantic_candidates,
    MODEL_NAME,
)
from repo import repo_embeddings_by_file


def ok(msg: str):
    print("OK:", msg)


def warn(msg: str):
    print("WARN:", msg)


def ensure_qa(file_hash: str, question: str) -> int:
    qn = normalize_question(question)
    qh = hash_question(qn)
    row = find_exact_qa(file_hash=file_hash, question_hash=qh)
    if row:
        return int(row["id"])
    return int(
        save_qa(
            file_hash=file_hash,
            question_original=question,
            question_norm=qn,
            question_hash=qh,
            answer=None,
            meta=None,
        )
    )


def main():
    with flask_app.app_context():
        file_a = "check-file-1"
        file_b = "other-file-iso"
        q1 = "Wie groß ist die Datei?"
        q2 = "Wie alt ist die Datei?"

        # 1) Create/reuse two QAs in the same file scope
        qa1 = ensure_qa(file_a, q1)
        qa2 = ensure_qa(file_a, q2)
        ok(f"QAs prepared in {file_a}: qa1={qa1}, qa2={qa2}")

        # 2) Embed + save for both; check upsert (double save)
        v1 = embed_question(normalize_question(q1))
        v2 = embed_question(normalize_question(q2))
        save_embedding(qa_id=qa1, vec=v1)
        save_embedding(qa_id=qa1, vec=v1)  # upsert second time
        save_embedding(qa_id=qa2, vec=v2)
        ok("Embeddings saved with upsert behavior (no errors).")

        # 3) Self-similarity should rank qa1 first for query v1
        cands = find_semantic_candidates(file_hash=file_a, query_vec=v1, k=5)
        print("Candidates for q1:", cands)
        if cands and cands[0][0] == qa1 and cands[0][1] > 0.95:
            ok("Self-similarity ≈ 1.0 and ranked first.")
        else:
            warn("Expected self-similarity near 1.0 as top candidate.")

        # 4) Negative example: qa2 score should be lower than qa1
        score_map = {qid: score for (qid, score) in cands}
        s1 = score_map.get(qa1, 0.0)
        s2 = score_map.get(qa2, 0.0)
        print(f"Scores → qa1: {s1:.4f}, qa2: {s2:.4f}")
        if s2 < s1:
            ok("Different question ranks lower than identical one.")
        else:
            warn("Unexpected: second question scored >= first.")

        # 5) File-hash isolation: save embedding for same vector under other file
        qa_other = ensure_qa(file_b, q1)
        save_embedding(qa_id=qa_other, vec=v1)
        rows_other = repo_embeddings_by_file(file_hash=file_b, model=MODEL_NAME)
        ok(f"Embeddings present for {file_b}: {len(rows_other)} row(s)")

        # Ensure candidates for file_a do not include qa_other
        ids_a = [qid for (qid, _score) in cands]
        if qa_other not in ids_a:
            ok("Isolation by file_hash confirmed (no cross-results).")
        else:
            warn("Isolation failed: candidate from other file appeared.")

        # 6) Bytes roundtrip: compare stored v1 with loaded v1 (tolerance)
        rows_a = repo_embeddings_by_file(file_hash=file_a, model=MODEL_NAME)
        loaded = None
        for r in rows_a:
            if int(r[0]) == qa1:
                dim = int(r[1])
                loaded = np.frombuffer(r[2], dtype=np.float32, count=dim)
                break
        if loaded is not None and loaded.shape[0] == v1.shape[0] and np.allclose(loaded, v1, atol=1e-5):
            ok("Bytes roundtrip successful (np.allclose).")
        else:
            warn("Bytes roundtrip mismatch for qa1 embedding.")

        print("\nChecks complete.")


if __name__ == "__main__":
    main()