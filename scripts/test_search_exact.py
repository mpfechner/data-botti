

# scripts/test_search_exact.py
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
    from services.qa_service import save_qa, make_query_request
    from services.search_service import SearchService
    from services.models import QARecord

    q_raw = "How many unique values are in column X?"
    file_hash = f"test-search-{uuid.uuid4().hex[:8]}"

    with app_ctx():
        # Build the QueryRequest using our helper
        req = make_query_request(q_raw, file_hash)

        # Ensure there's a row to find
        qa_id = save_qa(
            file_hash=file_hash,
            question_original=req.question_raw,
            question_norm=req.question_norm,
            question_hash=req.question_hash,
            answer=None,
            meta={"source": "test_search_exact"},
        )

        # Execute exact search via SearchService
        svc = SearchService()
        rec = svc.search_exact(req)

        assert isinstance(rec, QARecord), f"Expected QARecord, got {type(rec)}"
        assert rec.id == qa_id
        assert rec.question == q_raw
        assert rec.file_hash == file_hash

        print("OK ✅ SearchService.search_exact returned QARecord with correct values")


if __name__ == "__main__":
    main()