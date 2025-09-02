import os
import secrets


def get_secret_key() -> str:
    """Return the application's SECRET_KEY.
    Priority: environment variable, then instance file, else generate and persist."""
    # 1. Env var has priority
    env_key = os.getenv("SECRET_KEY")
    if env_key:
        return env_key

    # 2. Instance path file
    instance_path = os.path.join(os.getcwd(), "instance")
    os.makedirs(instance_path, exist_ok=True)
    key_path = os.path.join(instance_path, "secret_key")

    if os.path.exists(key_path):
        with open(key_path, "r") as f:
            return f.read().strip()

    # 3. Generate and persist
    new_key = secrets.token_hex(32)
    with open(key_path, "w") as f:
        f.write(new_key)
    try:
        os.chmod(key_path, 0o600)
    except Exception:
        pass
    return new_key


def get_config() -> dict:
    """Return app configuration settings."""
    return {
        "DEBUG": os.getenv("DEBUG", "false").lower() == "true",
        "UPLOAD_FOLDER": os.getenv("UPLOAD_FOLDER", "data"),
        "API_KEY": os.getenv("API_KEY", ""),
        "SECRET_KEY": get_secret_key(),
    }