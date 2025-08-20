CREATE DATABASE IF NOT EXISTS databotti;
USE databotti;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username TEXT NOT NULL,
    email TEXT NOT NULL
);

CREATE TABLE datasets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename TEXT NOT NULL,
    upload_date DATETIME NOT NULL,
    user_id INT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE prompts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    dataset_id INT,
    user_id INT,
    prompt_text TEXT,
    response_text TEXT,
    timestamp DATETIME NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE reports (
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