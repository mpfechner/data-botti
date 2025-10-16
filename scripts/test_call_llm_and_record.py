from services.qa_service import QaService


# scripts/test_call_llm_and_record.py
import sys, os
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
    from services.qa_service import make_query_request, call_llm_and_record

    q_raw = "Summarize the dataset in one sentence."
    file_hash = "test-llm-usage"

    with app_ctx():
        req = make_query_request(q_raw, file_hash)
        content = QaService.answer(req, model="gpt-4o-mini", max_tokens=50)

        print("[LLM Content]", content)
        if req.token_usage:
            print("[Token Usage]", req.token_usage)
        else:
            print("[Token Usage] None")


if __name__ == "__main__":
    main()