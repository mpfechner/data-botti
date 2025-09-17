CREATE DATABASE IF NOT EXISTS databotti;
USE databotti;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_admin TINYINT(1) NOT NULL DEFAULT 0,
    username VARCHAR(80) NULL,
    consent_given_at DATETIME NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_users_email (email)
);
-- ------------------------------------------------------------
-- Optional seed admin (commented by default)
-- How to use:
--   1) Generate a password hash locally (Python one-liner):
--        >>> from werkzeug.security import generate_password_hash
--        >>> print(generate_password_hash("ChangeMe123!"))
--      Copy the resulting hash.
--   2) Uncomment the INSERT below and replace email/username/hash as needed.
--      You can re-run the init or execute it manually against the DB.
--
-- INSERT INTO users (email, password_hash, username, is_admin)
-- VALUES ('admin@example.com', '<PASTE_HASH_HERE>', 'admin', 1);
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS groups (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(120) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_groups_name (name)
);

CREATE TABLE IF NOT EXISTS user_groups (
    user_id INT NOT NULL,
    group_id INT NOT NULL,
    added_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, group_id),
    KEY fk_user_groups_group (group_id),
    CONSTRAINT fk_user_groups_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_user_groups_group FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS datasets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename TEXT NOT NULL,
    upload_date DATETIME NOT NULL,
    user_id INT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS prompts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    dataset_id INT,
    user_id INT,
    prompt_text TEXT,
    response_text TEXT,
    timestamp DATETIME NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reports (
    id INT AUTO_INCREMENT PRIMARY KEY,
    dataset_id INT,
    summary TEXT,
    generated_on DATETIME NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

-- 1) Archiv-Metadaten zum Originalfile (Bytes bleiben auf Disk/Objektstore)
CREATE TABLE IF NOT EXISTS dataset_files (
    id INT AUTO_INCREMENT PRIMARY KEY,
    dataset_id INT NOT NULL UNIQUE,
    original_name TEXT NOT NULL,
    size_bytes BIGINT,
    file_hash CHAR(64) NOT NULL,
    encoding VARCHAR(40),
    delimiter VARCHAR(8),
    stored_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    file_path TEXT NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
    UNIQUE KEY uq_df_file_hash (file_hash)
);

-- 2) Spaltenprofil je Dataset
CREATE TABLE IF NOT EXISTS dataset_columns (
    id INT AUTO_INCREMENT PRIMARY KEY,
    dataset_id INT NOT NULL,
    ordinal INT NOT NULL,
    name VARCHAR(255) NOT NULL,
    dtype VARCHAR(40) NOT NULL,
    is_nullable TINYINT(1) NOT NULL DEFAULT 1,
    distinct_count BIGINT NULL,
    min_val TEXT NULL,
    max_val TEXT NULL,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
    UNIQUE KEY uq_dc_dataset_ordinal (dataset_id, ordinal),
    KEY idx_dc_dataset (dataset_id)
);

-- ============================================================
-- Q&A Pairs (Exact + Metadaten)
-- ============================================================
CREATE TABLE IF NOT EXISTS qa_pairs (
  id INT AUTO_INCREMENT PRIMARY KEY,
  file_hash VARCHAR(64) NOT NULL,                -- Scope (z. B. Datei-Hash)
  question_original TEXT NOT NULL,
  question_norm TEXT NOT NULL,                   -- normalisierte Frage
  question_hash CHAR(64) NOT NULL,               -- SHA-256 hex von question_norm
  answer TEXT NULL,
  meta JSON NULL,                                -- {source, tags, ...}
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_qa_question_hash_file (question_hash, file_hash),
  KEY idx_qa_file_created (file_hash, created_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Embeddings zu Q&A (distiluse-base-multilingual-cased-v2 → 512-D)
-- ============================================================
CREATE TABLE IF NOT EXISTS qa_embeddings (
  qa_id INT NOT NULL,
  model VARCHAR(80) NOT NULL,
  dim INT NOT NULL,
  vec BLOB NOT NULL,                             -- float32[dim] (z. B. 512*4 = 2048 B)
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (qa_id, model),
  CONSTRAINT fk_qaemb_qa FOREIGN KEY (qa_id)
    REFERENCES qa_pairs(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Token-Usage (LLM Telemetrie)
-- ============================================================
CREATE TABLE IF NOT EXISTS token_usage (
  id INT AUTO_INCREMENT PRIMARY KEY,
  ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  user_id INT NULL,
  model VARCHAR(80) NOT NULL,
  operation VARCHAR(40) NOT NULL,                -- z. B. 'qa.answer'
  prompt_tokens INT NOT NULL DEFAULT 0,
  completion_tokens INT NOT NULL DEFAULT 0,
  total_tokens INT NOT NULL DEFAULT 0,
  meta JSON NULL,
  KEY idx_tokenusage_ts (ts DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;