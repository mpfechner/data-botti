from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from app import app as flask_app
from services.embeddings import _load_model  # Test greift bewusst auf die interne Loader-Funktion zu


@pytest.mark.parametrize("_", [None])
def test_embedding_device_detects_accelerator(_):
    """Das Modell lädt und meldet ein gültiges Device (mps/cuda/cpu)."""
    with flask_app.app_context():
        _model, device = _load_model()
        assert device in {"mps", "cuda", "cpu"}


if __name__ == "__main__":
    # Manuell ausführbar, z. B. `python scripts/test_embedding_device.py`
    with flask_app.app_context():
        _model, device = _load_model()
        print(f"Embedding device: {device}")