# DataBotti

DataBotti is a data analyzing tool with AI support.  
Current status: **Stable MVP** – all major functions are implemented and stable, including AI functionality with semantic search based on embeddings and cross-encoder re-ranking.

---

## Important note on AI functionality
DataBotti requires a **paid API key for OpenAI** to enable its AI features.  
No API key is included or provided with this repository.  
You can obtain an API key by creating an account at [OpenAI](https://openai.com/api/).

---

---

## Features
- 📊 Database connection (MariaDB via Docker or locally with SQL script)  
- 🔍 Standard functions for data analysis  
- 🤖 Integrated AI-supported analyses with semantic search (embeddings and cross-encoder re-ranking)  
- 🛡️ Logging and safety-focused prompt design  
- 💾 Offline model caching and seed question management for QA  

---

## Setup

### 1. Clone the repository
```
git clone https://github.com/dein-user/data-botti.git
cd data-botti
```

### 2. Configure environment variables
Copy the example file:
```
cp .env-example .env
```
Edit `.env` and enter your own values (DB user, passwords, API key).

### 3. Prepare the database

**a) With Docker (recommended)**  
- The file `create_database.sql` must be located in **`docker/mysql/init/`** (relative to the project root). This folder is automatically mounted into the database container, so the script runs on first startup.  
- On the first start of the container, the script is executed automatically and the database is created.  
- Start the containers:
```
docker compose -f docker/docker-compose.yml up --build
```

**b) Without Docker (local)**  
- Install MariaDB/MySQL locally.  
- Run the script manually:  
```
mysql -u <user> -p < databotti < app/sql-scripts/create_database.sql
```

### 4. Start the backend
- With Docker, the backend runs directly as a service.  
- Alternatively, start locally (e.g., in PyCharm):
```
python app.py
```

---

## PyCharm workflow (local development)
For working in PyCharm there is an override file that only runs the DB in the container:
```
docker compose -f docker/docker-compose.yml -f docker/docker-compose.pycharm.yml up -d
```
Then you can start the backend locally in PyCharm while the database runs in the container.

---

## Release
- Version 1.12.0 – stable MVP with full AI semantic search, offline caching, and QA seed question management.

## License
- Licensed under CC BY-NC 4.0
