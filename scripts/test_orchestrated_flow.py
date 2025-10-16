

import os
import sys
# Ensure app root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contextlib import contextmanager

from services.qa_service import make_query_request
from services.search_service import SearchService


# Demo file hash constant
FILE_HASH = "orchestrated-demo"

# --- Flask app loader & context ---

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
    # Build one QA-like and one Analysis-like query
    queries = [
        "What is the capital of France?",
        "Zeige mir die 7 schwächsten Monate"
    ]

    with app_ctx():
        for question in queries:
            req = make_query_request(question, file_hash=FILE_HASH)
            rec = SearchService().search_orchestrated(req)
            print("=" * 40)
            print(f"Question: {question}")
            intent = getattr(req, "intent", None)
            decision = getattr(req, "decision", None)
            print(f"Detected intent: {intent}")
            print(f"Decision: {decision}")
            if rec is not None and getattr(rec, "answer", None):
                print(f"Result text: {rec.answer}")
            else:
                print("No direct result (analysis placeholder or will call LLM in app flow)")


if __name__ == "__main__":
    main()