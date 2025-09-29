import os
import sys
import unittest

# Ensure app path is available
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.search_service import SearchService
from services.models import QARecord
from services.qa_service import make_query_request
from app import app

# Additional imports for deterministic seeding and helpers
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from services.qa_service import make_query_request, save_qa, normalize_question
from services.embeddings import embed_passage


FILE_HASH = "semantic-test-file"
Q_ORIG = "What city is Germany's capital?"
A_TEXT = "Berlin is the capital of Germany."
MODEL_NAME = "intfloat/multilingual-e5-base"
LLM_FILE_HASH = "llm-test-file"

class TestSemanticFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ctx = app.app_context()
        cls._ctx.push()

        cls.engine = app.config.get("DB_ENGINE")
        if cls.engine is None:
            raise RuntimeError("DB engine not configured on app for tests")

        # Ensure a QA exists for our FILE_HASH; insert-or-get by (file_hash, question_hash)
        req = make_query_request(Q_ORIG, FILE_HASH)
        try:
            qa_id = save_qa(
                file_hash=FILE_HASH,
                question_original=Q_ORIG,
                question_norm=normalize_question(Q_ORIG),
                question_hash=req.question_hash,
                answer=A_TEXT,
            )
            cls.qa_id = int(qa_id)
        except IntegrityError:
            # Already exists → fetch id
            with cls.engine.begin() as conn:
                row = conn.execute(
                    text("SELECT id FROM qa_pairs WHERE file_hash=:fh AND question_hash=:qh LIMIT 1"),
                    {"fh": FILE_HASH, "qh": req.question_hash},
                ).mappings().first()
                cls.qa_id = int(row["id"]) if row else None
        if not getattr(cls, "qa_id", None):
            raise RuntimeError("Failed to ensure QA row for semantic test")

        # Ensure an embedding exists in qa_embeddings for this qa_id and model
        vec = embed_passage(normalize_question(Q_ORIG))
        import numpy as _np
        vec_np = _np.asarray(vec, dtype=_np.float32)
        dim = int(vec_np.shape[0])
        vec_bytes = vec_np.tobytes()
        with cls.engine.begin() as conn:
            try:
                conn.execute(
                    text(
                        """
                        INSERT INTO qa_embeddings (qa_id, model, dim, vec)
                        VALUES (:qa_id, :model, :dim, :vec)
                        """
                    ),
                    {"qa_id": cls.qa_id, "model": MODEL_NAME, "dim": dim, "vec": vec_bytes},
                )
            except IntegrityError:
                # Already present → ignore
                pass

    @classmethod
    def tearDownClass(cls):
        # Pop app context after all tests
        cls._ctx.pop()

    def setUp(self):
        self.service = SearchService()

    def test_semantic_fallback_when_no_exact_match(self):
        # Query that should not hit exact, forcing semantic fallback
        req = make_query_request("Capital of Germany?", file_hash=FILE_HASH)
        result = self.service.search_orchestrated(req)
        self.assertEqual(req.decision, "semantic")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, QARecord)

    def test_exact_match_when_available(self):
        # Query that mimics exact stored Q/A
        req = make_query_request(Q_ORIG, file_hash=FILE_HASH)
        result = self.service.search_orchestrated(req)
        self.assertEqual(req.decision, "exact")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, QARecord)


    def test_llm_fallback_when_no_match(self):
        """When there is no exact and no semantic candidate, orchestrator should signal LLM."""
        # Use a file_hash with no seeded QAs/embeddings to force no candidates
        req = make_query_request("What's the weather on Mars?", file_hash=LLM_FILE_HASH)
        result = self.service.search_orchestrated(req)
        self.assertEqual(req.decision, "llm")
        # No QARecord expected at this stage; LLM happens in the upper layer
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()