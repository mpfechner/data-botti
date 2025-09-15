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
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    app.logger.setLevel(level)
    app.logger.propagate = False

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
        fh.setFormatter(fmt)
        fh.setLevel(level)
        app.logger.addHandler(fh)

    # Route non-app loggers to the same file, silence default console output
    root = logging.getLogger()
    # Remove default StreamHandler(s) that print to CLI
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)
    # Attach rotating file handler to root as well
    root_fh = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    root_fh.setFormatter(fmt)
    root_fh.setLevel(level)
    root.addHandler(root_fh)

    # Reduce noisy werkzeug logs to WARNING and prevent propagation to console
    wlog = logging.getLogger("werkzeug")
    for h in list(wlog.handlers):
        wlog.removeHandler(h)
    wlog.setLevel(logging.WARNING)
    wlog.propagate = False