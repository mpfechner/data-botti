

import sys, os, uuid
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
    from services.qa_service import make_query_request, save_qa, orchestrate
    from services.models import QARecord, MatchResults

    file_hash = f"test-orchestrate-{uuid.uuid4().hex[:8]}"

    with app_ctx():
        # --- Exact case ---
        question1 = "What is the first column?"
        q1 = make_query_request(question1, file_hash)
        qa_id = save_qa(
            file_hash=file_hash,
            question_original=q1.question_raw,
            question_norm=q1.question_norm,
            question_hash=q1.question_hash,
            answer="This is a mock answer for column 1.",
            meta={"source": "test_orchestrate"},
        )

        req_exact = make_query_request(question1, file_hash)
        result_exact = orchestrate(req_exact)
        assert isinstance(result_exact, MatchResults)
        assert result_exact.mode == "exact"
        assert req_exact.decision == "exact"
        print("[Exact] decision=", req_exact.decision, "records=", len(result_exact.records))

        # --- LLM case ---
        question2 = "Summarize the dataset briefly."
        req_llm = make_query_request(question2, file_hash)
        result_llm = orchestrate(req_llm)
        assert isinstance(result_llm, MatchResults)
        assert result_llm.mode == "llm"
        assert req_llm.decision == "llm"
        assert result_llm.records and isinstance(result_llm.records[0], QARecord)
        print("[LLM] decision=", req_llm.decision, "records=", len(result_llm.records))


if __name__ == "__main__":
    main()