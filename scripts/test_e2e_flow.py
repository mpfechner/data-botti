import os
import os as _os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app
from services.qa_service import make_query_request, save_qa, normalize_question
from services.search_service import SearchService
# Additional imports for DB setup
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

# Test constants
TEST_USER_ID = 1
TEST_FILENAME = "e2e_test_dataset.csv"
TEST_FILE_HASH = "e2e-hash-0001"

# Helper: pick an existing dataset+file; otherwise create one that matches schema
def ensure_test_dataset(engine):
    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT d.id AS dataset_id, df.file_hash, d.user_id AS owner_id
                FROM datasets d
                JOIN dataset_files df ON df.dataset_id = d.id
                ORDER BY d.upload_date DESC, d.id DESC
                LIMIT 1
                """
            )
        ).mappings().first()
        if row:
            return int(row["dataset_id"]), row["file_hash"], int(row["owner_id"]) if row.get("owner_id") is not None else TEST_USER_ID

        # None found → create a minimal dataset owned by TEST_USER_ID
        conn.execute(
            text(
                """
                INSERT INTO datasets (filename, user_id, upload_date)
                VALUES (:fn, :uid, NOW())
                """
            ),
            {"fn": TEST_FILENAME, "uid": TEST_USER_ID},
        )
        ds_row = conn.execute(text("SELECT LAST_INSERT_ID() AS id")).mappings().first()
        ds_id = int(ds_row["id"]) if ds_row else None
        if not ds_id:
            raise RuntimeError("Failed to create test dataset")

        # Generate a unique 64-char hex file hash
        gen_hash = _os.urandom(32).hex()

        # Create dataset_files row with required/not-null fields from your schema
        conn.execute(
            text(
                """
                INSERT INTO dataset_files (
                    dataset_id, original_name, size_bytes, file_hash, encoding, delimiter, stored_at, file_path
                )
                VALUES (:dsid, :orig, :size, :fh, :enc, :delim, NOW(), NULL)
                """
            ),
            {
                "dsid": ds_id,
                "orig": TEST_FILENAME,
                "size": 0,
                "fh": gen_hash,
                "enc": "utf-8",
                "delim": ",",
            },
        )
        return ds_id, gen_hash, TEST_USER_ID


# Insert or fetch existing QA for (file_hash, question)
def ensure_qa(engine, file_hash: str, question: str, answer: str) -> int:
    req = make_query_request(question, file_hash)
    try:
        qa_id = save_qa(
            file_hash=file_hash,
            question_original=question,
            question_norm=normalize_question(question),
            question_hash=req.question_hash,
            answer=answer,
        )
        return int(qa_id)
    except IntegrityError:
        # Already exists → look it up
        with engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT id FROM qa_pairs
                    WHERE file_hash = :fh AND question_hash = :qh
                    LIMIT 1
                    """
                ),
                {"fh": file_hash, "qh": req.question_hash},
            ).mappings().first()
            if not row:
                raise
            return int(row["id"])


class TestEndToEndFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.app_context = app.app_context()
        cls.app_context.push()
        # Setup test dataset and file
        cls.engine = app.config.get("DB_ENGINE")
        cls.dataset_id, cls.file_hash, cls.owner_user_id = ensure_test_dataset(cls.engine)

    def test_e2e_exact_flow(self):
        qa_id = ensure_qa(self.engine, self.file_hash, "What is the capital of France?", "Paris is the capital of France.")
        req = make_query_request("What is the capital of France?", self.file_hash)
        SearchService().search_orchestrated(req)
        self.assertEqual(req.decision, "exact")

    def test_e2e_analysis_flow(self):
        query_request = make_query_request("Show me a chart of sales trends for 2023", self.file_hash)
        SearchService().search_orchestrated(query_request)
        self.assertEqual(query_request.decision, "analysis")

    def test_e2e_routes(self):
        qa_id = ensure_qa(self.engine, self.file_hash, "What is the capital of France?", "Paris is the capital of France.")
        # Set session user_id for access
        with self.client.session_transaction() as sess:
            sess["user_id"] = self.owner_user_id
        response = self.client.get(f"/search?dataset_id={self.dataset_id}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("What is the capital of France?", response.get_data(as_text=True))

        with self.client.session_transaction() as sess:
            sess["user_id"] = self.owner_user_id
        response = self.client.get(f"/qa/{qa_id}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.get_data(as_text=True))

if __name__ == "__main__":
    unittest.main()
