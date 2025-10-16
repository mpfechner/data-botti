

import os
from sqlalchemy import create_engine


def init_engine():
    """Create and return a SQLAlchemy engine using environment variables."""
    user = os.getenv("MYSQL_USER")
    pwd = os.getenv("MYSQL_PASSWORD")
    db = os.getenv("MYSQL_DATABASE")
    host = os.getenv("MYSQL_HOST", "127.0.0.1")
    port = os.getenv("MYSQL_PORT", "3306")

    engine = create_engine(
        f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}",
        pool_pre_ping=True,  # prüft Connection vor Nutzung → verhindert 2013-Fehler
        future=True,
    )
    return engine

_engine_instance = None

def get_engine():
    """Return cached SQLAlchemy engine or create if not yet initialized."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = init_engine()
    return _engine_instance
