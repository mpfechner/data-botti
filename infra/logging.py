import logging
from logging.handlers import RotatingFileHandler
import os

# Helper to get log level from environment variable
def _get_level_from_env(default: int = logging.DEBUG) -> int:
    level_name = os.getenv("LOG_LEVEL", "").upper()
    if not level_name:
        return default
    return getattr(logging, level_name, default)

def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with rotating file handler.
    Respects LOG_DIR and LOG_LEVEL env vars. Prevents duplicate handlers.
    """
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "databotti.log")

    logger = logging.getLogger(name)
    logger.propagate = False
    level = _get_level_from_env(logging.DEBUG)
    logger.setLevel(level)

    # Avoid duplicate handlers for the same file
    need_handler = True
    for h in logger.handlers:
        if isinstance(h, RotatingFileHandler):
            try:
                if os.path.samefile(getattr(h, "baseFilename", ""), log_path):
                    need_handler = False
                    break
            except Exception:
                pass

    if need_handler:
        handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)

    return logger


# Attach rotating file handler to Flask's app.logger
def setup_app_logging(app) -> None:
    """Attach rotating file handler to Flask's app.logger and set level.
    Respects LOG_DIR and LOG_LEVEL.
    """
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "databotti.log")

    level = _get_level_from_env(logging.DEBUG)
    app.logger.setLevel(level)

    # Avoid duplicate handlers
    need_handler = True
    for h in app.logger.handlers:
        if isinstance(h, RotatingFileHandler):
            try:
                if os.path.samefile(getattr(h, "baseFilename", ""), log_path):
                    need_handler = False
                    break
            except Exception:
                pass

    if need_handler:
        fh = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        fh.setFormatter(fmt)
        fh.setLevel(level)
        app.logger.addHandler(fh)