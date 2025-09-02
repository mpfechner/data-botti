import os

def get_config() -> dict:
    """Return app configuration settings."""
    return {
        "DEBUG": os.getenv("DEBUG", "false").lower() == "true",
        "UPLOAD_FOLDER": os.getenv("UPLOAD_FOLDER", "data"),
        "API_KEY": os.getenv("API_KEY", ""),
    }