import logging
from logging.handlers import RotatingFileHandler
import os

def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with rotating file handler."""
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "databotti.log")

    logger = logging.getLogger(name)
    if not logger.handlers:  # prevent duplicate handlers
        handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger