import os
import sys
import unittest
import uuid
import pandas as pd

# Ensure app path is available
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy import text

from app import app
from services.qa_service import make_query_request, normalize_question


TEST_USER_ID = 1


def ensure_dataset_with_file(engine):
    """Always create a fresh dataset+file for isolation; return (dataset_id, file_hash, owner_id)."""
    with engine.begin() as conn:
        # Fresh dataset owned by test user; unique filename for clarity
        ds_name = f"llm_fallback_test_{os.getpid()}_{uuid.uuid4().hex[:6]}.csv"

        conn.execute(
            text(
                """
                INSERT INTO datasets (filename, user_id, upload_date)
                VALUES (:fn, :uid, NOW())
                """
            ),
            {"fn": ds_name, "uid": TEST_USER_ID},
        )
        ds_id = conn.execute(text("SELECT LAST_INSERT_ID() AS id")).mappings().first()["id"]

        # Create a real CSV file on disk so the route can load it
        file_path = f"/tmp/{ds_name}"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("date,value,city\n")
            f.write("2024-05-01,42,Berlin\n")
            f.write("2024-05-02,123,Hamburg\n")
            f.write("2024-05-03,99,Munich\n")
        size_bytes = os.path.getsize(file_path)

        # Fresh dataset_files entry with required fields; 64-hex file_hash
        gen_hash = uuid.uuid4().hex + uuid.uuid4().hex
        conn.execute(
            text(
                """
                INSERT INTO dataset_files (
                    dataset_id, original_name, size_bytes, file_hash, encoding, delimiter, stored_at, file_path
                )
                VALUES (:dsid, :orig, :size, :fh, :enc, :delim, NOW(), :path)
                """
            ),
            {
                "dsid": ds_id,
                "orig": ds_name,
                "size": size_bytes,
                "fh": gen_hash,
                "enc": "utf-8",
                "delim": ",",
                "path": file_path,
            },
        )
        return int(ds_id), gen_hash, TEST_USER_ID


class TestLLMFallbackRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["WTF_CSRF_ENABLED"] = False  # simplify POST in tests
        cls._ctx = app.app_context()
        cls._ctx.push()
        cls.engine = app.config.get("DB_ENGINE")
        if cls.engine is None:
            raise RuntimeError("DB engine not configured on app for tests")
        cls.dataset_id, cls.file_hash, cls.owner_user_id = ensure_dataset_with_file(cls.engine)

    @classmethod
    def tearDownClass(cls):
        cls._ctx.pop()

    def setUp(self):
        self.client = app.test_client()
        # ensure logged in as dataset owner
        with self.client.session_transaction() as sess:
            sess["user_id"] = self.owner_user_id

        # Monkeypatch at the module actually used by the route (routes.assistant)
        import routes.assistant as assistant_module
        self._assistant_module = assistant_module

        self._orig_call_model = assistant_module.call_model
        self._orig_select_cols = assistant_module.select_relevant_columns
        self._orig_load_csv = assistant_module.load_csv_resilient

        assistant_module.call_model = lambda **kwargs: ("[MOCK LLM ANSWER] Deterministic.", None)
        assistant_module.select_relevant_columns = lambda *args, **kwargs: []
        assistant_module.load_csv_resilient = lambda *args, **kwargs: (
            pd.DataFrame({
                "date": ["2024-05-01", "2024-05-02", "2024-05-03"],
                "value": [42, 123, 99],
                "city": ["Berlin", "Hamburg", "Munich"],
            }),
            "utf-8",
            ",",
        )

    def tearDown(self):
        # restore patches
        self._assistant_module.call_model = self._orig_call_model
        self._assistant_module.select_relevant_columns = self._orig_select_cols
        self._assistant_module.load_csv_resilient = self._orig_load_csv

    def test_llm_fallback_creates_qa_and_renders_answer(self):
        # Use a prompt that shouldn't match any exact/semantic entry in this dataset
        prompt = f"Compose a limerick about purple elephants (test-id={uuid.uuid4().hex[:8]})"

        # POST to the AI prompt route
        # First, accept AI consent (route expects this flag)
        self.client.post(f"/ai/{self.dataset_id}", data={"consent": "1"}, follow_redirects=True)

        # Now send the actual prompt
        resp = self.client.post(f"/ai/{self.dataset_id}", data={"prompt": prompt}, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("[MOCK LLM ANSWER] Deterministic.", html)

        # The route chooses the file_hash internally (latest file for the dataset). Look up by question_hash alone.
        qh = make_query_request(prompt, self.file_hash).question_hash
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT id, answer, file_hash FROM qa_pairs
                    WHERE question_hash = :qh
                    ORDER BY id DESC LIMIT 1
                    """
                ),
                {"qh": qh},
            ).mappings().first()
        self.assertIsNotNone(row)
        self.assertIn("MOCK LLM ANSWER", row["answer"])  # stored answer contains our mock

    def test_llm_fallback_reuses_existing_qa(self):
        """Ensure repeated identical prompts do not create duplicate QA entries."""
        prompt = "Please summarize the dataset columns in one paragraph."

        # Accept AI consent first (same as the create test)
        self.client.post(f"/ai/{self.dataset_id}", data={"consent": "1"}, follow_redirects=True)

        # Compute question_hash using the actual file used by the route (latest file for this dataset)
        with self.engine.begin() as conn:
            used_fh = conn.execute(
                text("SELECT file_hash FROM dataset_files WHERE dataset_id = :dsid ORDER BY stored_at DESC LIMIT 1"),
                {"dsid": self.dataset_id},
            ).scalar()
        qh = make_query_request(prompt, used_fh).question_hash

        # 1st call – should create a QA entry
        resp1 = self.client.post(
            f"/ai/{self.dataset_id}", data={"prompt": prompt}, follow_redirects=True
        )
        self.assertEqual(resp1.status_code, 200)

        # Count QA rows for this question_hash after first call
        with self.engine.begin() as conn:
            cnt1 = conn.execute(
                text("SELECT COUNT(*) AS c FROM qa_pairs WHERE question_hash = :qh AND file_hash = :fh"),
                {"qh": qh, "fh": used_fh},
            ).scalar()
        self.assertEqual(cnt1, 1, "Expected exactly one QA row after first LLM fallback call")

        # 2nd call – should reuse the same QA (no new row)
        resp2 = self.client.post(
            f"/ai/{self.dataset_id}", data={"prompt": prompt}, follow_redirects=True
        )
        self.assertEqual(resp2.status_code, 200)

        with self.engine.begin() as conn:
            cnt2 = conn.execute(
                text("SELECT COUNT(*) AS c FROM qa_pairs WHERE question_hash = :qh AND file_hash = :fh"),
                {"qh": qh, "fh": used_fh},
            ).scalar()
        self.assertEqual(
            cnt2, 1, "Expected reuse of existing QA entry; duplicate was created"
        )


if __name__ == "__main__":
    unittest.main()