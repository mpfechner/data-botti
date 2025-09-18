# scripts/test_qas_dataclasses.py
import sys, uuid
import os
# Ensure we can import from the app root (this file lives in app/scripts/)
_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.dirname(_HERE)
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)
from contextlib import contextmanager

# App-Kontext laden (funktioniert mit entweder app- oder factory-Setup)
def _load_app():
    try:
        from app import app as flask_app
        return flask_app
    except Exception:
        from app import create_app  # type: ignore
        return create_app()

@contextmanager
def app_ctx():
    app = _load_app()
    ctx = app.app_context()
    ctx.push()
    try:
        yield
    finally:
        ctx.pop()

def main():
    from services.qa_service import (
        make_query_request,
        save_qa,
        find_exact_qa,
        embed_question,
        save_embedding,
    )
    from services.models import QARecord

    question_raw = sys.argv[1] if len(sys.argv) > 1 else "How many rows are in the dataset?"
    file_hash = sys.argv[2] if len(sys.argv) > 2 else f"demo-file-{uuid.uuid4().hex[:8]}"

    with app_ctx():
        q = make_query_request(question_raw, file_hash)
        qa_id = save_qa(
            file_hash=file_hash,
            question_original=q.question_raw,
            question_norm=q.question_norm,
            question_hash=q.question_hash,
            answer=None,
            meta={"source": "test_qas_dataclasses"},
        )
        print(f"[insert] qa_id={qa_id}")

        rec = find_exact_qa(file_hash=file_hash, question_hash=q.question_hash)
        assert rec is not None, "Exact find returned None"
        print(f"[find] type={type(rec).__name__}")

        # Tolerant: akzeptiert sowohl Mapping als auch QARecord
        if isinstance(rec, QARecord):
            print(f"[find] QARecord(id={rec.id}, question='{rec.question}', file_hash='{rec.file_hash}')")
            text_norm = rec.question_norm
            qa_id_for_embed = rec.id or qa_id
        else:
            print(f"[find] row.keys={list(rec.keys())}")
            text_norm = rec.get("question_norm") or q.question_norm
            qa_id_for_embed = rec.get("id") or qa_id

        vec = embed_question(text_norm)
        save_embedding(qa_id_for_embed, vec)
        print(f"[embed] saved for qa_id={qa_id_for_embed}, dim={len(vec)}")

        print("OK ✅ Insert → FindExact → Embedding")

if __name__ == "__main__":
    main()