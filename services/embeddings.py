

import logging
from typing import List
from threading import Lock

import torch
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "intfloat/multilingual-e5-base"
_model = None
_device = None
_lock = Lock()

def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def _load_model():
    global _model, _device
    with _lock:
        if _model is None:
            _device = _get_device()
            logging.info(f"Loading model '{_MODEL_NAME}' on device '{_device}'")
            _model = SentenceTransformer(_MODEL_NAME, device=_device)
    return _model, _device

def embed_query(text: str) -> List[float]:
    """
    Embeds a query string using the E5 model, with 'query: ' prefix.
    """
    model, _ = _load_model()
    input_text = f"query: {text}"
    emb = model.encode(input_text, show_progress_bar=False, convert_to_numpy=True)
    return emb.tolist()

def embed_passage(text: str) -> List[float]:
    """
    Embeds a passage string using the E5 model, with 'passage: ' prefix.
    """
    model, _ = _load_model()
    input_text = f"passage: {text}"
    emb = model.encode(input_text, show_progress_bar=False, convert_to_numpy=True)
    return emb.tolist()