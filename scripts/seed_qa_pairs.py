import os
import hashlib
from sqlalchemy import create_engine, text

# --- Database connection setup ---
DB_USER = os.getenv("MYSQL_USER", "michael")
DB_PASS = os.getenv("MYSQL_PASSWORD", "test123")
DB_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
DB_NAME = os.getenv("MYSQL_DATABASE", "databotti")
DB_PORT = os.getenv("MYSQL_PORT", "3306")

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}?charset=utf8mb4"
)

# --- Example seed dataset (extend later) ---
SEED_QA_PAIRS = [
    {
        "question": "Was ist eine CSV-Datei?",
        "answer": "Eine CSV-Datei (Comma Separated Values) speichert tabellarische Daten, bei denen jede Zeile einen Datensatz darstellt und die Spalten durch Trennzeichen (meist Komma) getrennt sind.",
    },
    {
        "question": "Wie erkennt man den Delimiter einer CSV-Datei?",
        "answer": "Typische Delimiter sind Komma, Semikolon, Tabulator oder Pipe (|). Man erkennt ihn meist in der ersten Zeile oder kann ihn automatisch ermitteln, z. B. mit Python csv.Sniffer().",
    },
]

def seed_qa_pairs():
    with engine.begin() as conn:
        inserted, skipped = 0, 0
        for pair in SEED_QA_PAIRS:
            q = pair["question"].strip()
            a = pair["answer"].strip()
            q_hash = hashlib.sha256(q.encode("utf-8")).hexdigest()

            check_sql = text("SELECT COUNT(*) FROM qa_pairs WHERE question_hash=:qh AND is_seed=1")
            exists = conn.execute(check_sql, {"qh": q_hash}).scalar()

            if exists:
                skipped += 1
                continue

            insert_sql = text("""
                INSERT INTO qa_pairs (file_hash, question_original, question_norm, question_hash, answer, meta, is_seed)
                VALUES (:fh, :qo, :qn, :qh, :a, JSON_OBJECT('source','seed'), 1)
            """)
            conn.execute(insert_sql, {
                "fh": "seed_data",
                "qo": q,
                "qn": q.lower(),
                "qh": q_hash,
                "a": a
            })
            inserted += 1

        print(f"Seed complete: {inserted} inserted, {skipped} skipped.")

if __name__ == "__main__":
    seed_qa_pairs()
