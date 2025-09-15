CREATE DATABASE IF NOT EXISTS databotti;
USE databotti;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    username VARCHAR(80) NULL,
    consent_given_at DATETIME NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_users_email (email)
);

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
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE
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