# scripts/test_find_exact_qa.py
import sys, uuid, os
from contextlib import contextmanager

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.dirname(_HERE)
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

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
    from services.qa_service import save_qa, make_query_request, find_exact_qa
    from services.models import QARecord

    q_raw = "What is the first column?"
    file_hash = f"test-find-{uuid.uuid4().hex[:8]}"

    with app_ctx():
        q = make_query_request(q_raw, file_hash)
        qa_id = save_qa(
            file_hash=file_hash,
            question_original=q.question_raw,
            question_norm=q.question_norm,
            question_hash=q.question_hash,
            meta={"source": "test_find_exact_qa"},
        )
        rec = find_exact_qa(file_hash, q.question_hash)
        assert isinstance(rec, QARecord), f"Expected QARecord, got {type(rec)}"
        assert rec.id == qa_id
        assert rec.question == q_raw
        assert rec.file_hash == file_hash
        print("OK ✅ find_exact_qa returned QARecord with correct values")

if __name__ == "__main__":
    main()