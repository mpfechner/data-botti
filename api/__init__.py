from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Iterable, Optional

from flask import Blueprint, jsonify, request, current_app, g

from sqlalchemy import text
from infra.auth_helpers import api_auth_required, current_api_user_id, issue_api_token

from time import perf_counter
from sqlalchemy import text as _text
from repo import repo_qa_find_by_hash, repo_qa_find_by_id
from services.qa_service import make_query_request, embed_query, find_semantic_candidates

# -----------------------------------------------------------------------------
# Blueprint
# -----------------------------------------------------------------------------
api_v1 = Blueprint("api_v1", __name__, url_prefix="/api/v1")

# -----------------------------------------------------------------------------
# Helpers: JSON responses
# -----------------------------------------------------------------------------
def json_ok(payload: dict | None = None, status: int = 200):
  """
  Success response envelope.
  Returns: (flask.Response, status_code)
  """
  body: dict[str, Any] = {"ok": True}
  if payload:
    body.update(payload)
  return jsonify(body), status


def json_error(message: str, code: int = 400, **extra):
  """
  Error response envelope.
  Returns: (flask.Response, status_code)
  """
  body: dict[str, Any] = {"ok": False, "error": message}
  if extra:
    body.update(extra)
  return jsonify(body), code


# -----------------------------------------------------------------------------
# Auth: simple Bearer token check (for API calls, not browser sessions)
# -----------------------------------------------------------------------------
def _extract_bearer_token(auth_header: str | None) -> Optional[str]:
  if not auth_header:
    return None
  if not auth_header.lower().startswith("bearer "):
    return None
  return auth_header.split(" ", 1)[1].strip() or None


def require_api_auth(optional: bool = False):
  """
  Decorator enforcing Bearer token authentication for API endpoints.

  - Reads valid tokens from `current_app.config["API_TOKENS"]`
    (iterable of strings). Empty/None means: no tokens configured.
  - If `optional=True`, lets requests pass without a token but still
    attaches `g.api_token` if a valid token was provided.
  """
  def _decorator(fn: Callable):
    @wraps(fn)
    def _wrapped(*args, **kwargs):
      token = _extract_bearer_token(request.headers.get("Authorization"))
      valid_tokens: Iterable[str] = current_app.config.get("API_TOKENS") or []
      is_valid = token is not None and token in valid_tokens

      if is_valid:
        g.api_token = token  # may be useful in handlers

      if not is_valid and not optional:
        return json_error("Unauthorized", 401)

      return fn(*args, **kwargs)
    return _wrapped
  return _decorator


# -----------------------------------------------------------------------------
# Example endpoints
# -----------------------------------------------------------------------------
@api_v1.get("/ping")
def ping():
  """
  Lightweight health check (no auth).
  """
  return json_ok({"message": "pong"})


@api_v1.get("/whoami")
@api_auth_required
def whoami():
  """
  Requires either a Bearer token or a logged-in session.
  Returns the current user's ID if authenticated.
  """
  user_id = current_api_user_id()
  if user_id is None:
    return json_error("Unauthorized", 401)
  return json_ok({"user_id": user_id})


# -----------------------------------------------------------------------------
# API token issuance endpoint
# -----------------------------------------------------------------------------
@api_v1.post("/token")
def create_token():
  """
  Issue a short-lived API token for programmatic access.

  Accepts either JSON **or** form-encoded bodies.

  JSON body example:
    { "email": "...", "password": "..." }

  Returns:
    { "ok": true, "token": "..." }
  """
  # Accept JSON or form data
  data_json = request.get_json(silent=True) or {}
  data_form = request.form or {}

  # Prefer JSON, then fall back to form fields
  email = (data_json.get("email") or data_form.get("email") or "").strip().lower()
  password = (data_json.get("password") or data_form.get("password") or "")

  if not email or not password:
    return json_error("Missing 'email' or 'password'.", 400)

  try:
    token = issue_api_token(email=email, password=password)
  except ValueError as e:
    # Credential problems or explicit rejections from helper
    current_app.logger.warning("Token issuance rejected for %s: %s", email, e)
    return json_error("Invalid credentials.", 401)
  except Exception:
    # Unexpected internal failure – keep message generic for security
    current_app.logger.exception("Token issuance failed for %s", email)
    return json_error("Internal error during token issuance.", 500)

  if not token:
    # Helper returned falsy without raising – treat as invalid creds
    return json_error("Invalid credentials.", 401)

  return json_ok({"token": token}, 200)


# -----------------------------------------------------------------------------
# List datasets endpoint
# -----------------------------------------------------------------------------
@api_v1.get("/datasets")
@api_auth_required
def list_datasets():
  """
  Return datasets visible to the authenticated user.
  Visibility rule:
    - Owner (datasets.user_id = current user), OR
    - Member of a group that has access via datasets_groups.

  Response:
    { "ok": true, "items": [ { "id": ..., "filename": "...", "upload_date": "YYYY-MM-DD HH:MM:SS" }, ... ] }
  """
  user_id = current_api_user_id()
  if user_id is None:
    return json_error("Unauthorized", 401)

  engine = current_app.config.get("DB_ENGINE")
  if engine is None:
    return json_error("Database engine not configured.", 500)

  sql = text("""
    SELECT DISTINCT d.id, d.filename, d.upload_date
    FROM datasets d
    LEFT JOIN datasets_groups dg ON dg.dataset_id = d.id
    LEFT JOIN user_groups ug ON ug.group_id = dg.group_id
    WHERE d.user_id = :uid OR ug.user_id = :uid
    ORDER BY d.upload_date DESC, d.id DESC
  """)

  items = []
  try:
    with engine.begin() as conn:
      for row in conn.execute(sql, {"uid": user_id}):
        items.append({
          "id": int(row[0]),
          "filename": row[1],
          "upload_date": row[2].isoformat() if hasattr(row[2], "isoformat") else str(row[2]),
        })
  except Exception as e:
    current_app.logger.exception("Failed to list datasets via API")
    return json_error("Failed to load datasets.", 500)

  return json_ok({"items": items}, 200)

# -----------------------------------------------------------------------------
# Search-only endpoint over existing QA pairs (Exact → Semantic; no LLM)
# -----------------------------------------------------------------------------
@api_v1.post("/search")
@api_auth_required
def api_search_only():
  user_id = current_api_user_id()
  if user_id is None:
    return json_error("Unauthorized", 401)

  # Accept JSON or form data
  data_json = request.get_json(silent=True) or {}
  data_form = request.form or {}
  prompt = (data_json.get("prompt") or data_form.get("prompt") or "").strip()
  dataset_id = data_json.get("dataset_id") or data_form.get("dataset_id")
  try:
    dataset_id = int(dataset_id)
  except Exception:
    dataset_id = None
  top_k = data_json.get("top_k") or data_form.get("top_k") or 3
  try:
    top_k = max(1, min(10, int(top_k)))
  except Exception:
    top_k = 3

  if not prompt or not dataset_id:
    return json_error("Missing 'prompt' or 'dataset_id'.", 400)

  engine = current_app.config.get("DB_ENGINE")
  if engine is None:
    return json_error("Database engine not configured.", 500)

  # Resolve file_hash for this dataset with access control (owner or group member)
  sql = _text(
    """
    SELECT df.file_hash
    FROM datasets d
    JOIN dataset_files df ON df.dataset_id = d.id
    LEFT JOIN datasets_groups dg ON dg.dataset_id = d.id
    LEFT JOIN user_groups ug ON ug.group_id = dg.group_id
    WHERE d.id = :did AND (d.user_id = :uid OR ug.user_id = :uid)
    LIMIT 1
    """
  )
  file_hash = None
  try:
    with engine.begin() as conn:
      row = conn.execute(sql, {"did": dataset_id, "uid": user_id}).first()
      if row:
        file_hash = row[0]
  except Exception:
    current_app.logger.exception("Failed to resolve file_hash for dataset_id=%s", dataset_id)
    return json_error("Failed to resolve dataset file.", 500)

  if not file_hash:
    return json_error("Dataset not found or access denied.", 404)

  t0 = perf_counter()
  # Build normalized request (yields question_norm + question_hash)
  req = make_query_request(prompt, file_hash)

  # 1) Exact
  try:
    rec = repo_qa_find_by_hash(file_hash=file_hash, question_hash=req.question_hash)
  except Exception:
    current_app.logger.exception("Exact lookup failed for dataset_id=%s", dataset_id)
    return json_error("Exact search failed.", 500)

  if rec is not None:
    took_ms = (perf_counter() - t0) * 1000.0
    return json_ok({
      "found": True,
      "decision": "exact",
      "dataset_id": dataset_id,
      "file_hash": file_hash,
      "took_ms": round(took_ms, 2),
      "records": [{
        "qa_id": rec.id,
        "question": rec.question_original or rec.question_norm,
        "answer": rec.answer,
        "score": 1.0,
        "badge": "exact",
        "file_hash": rec.file_hash,
      }],
    })

  # 2) Semantic
  try:
    q_vec = embed_query(req.question_norm)
    candidates = find_semantic_candidates(file_hash, q_vec)
  except Exception:
    current_app.logger.exception("Semantic search failed for dataset_id=%s", dataset_id)
    return json_error("Semantic search failed.", 500)

  records = []
  THRESHOLD = 0.75
  for qa_id, score in (candidates or [])[:top_k]:
    if score < THRESHOLD:
      break
    rec2 = repo_qa_find_by_id(qa_id=qa_id)
    if not rec2:
      continue
    records.append({
      "qa_id": rec2.id,
      "question": rec2.question_original or rec2.question_norm,
      "answer": rec2.answer,
      "score": float(score),
      "badge": "semantic",
      "file_hash": rec2.file_hash,
    })

  took_ms = (perf_counter() - t0) * 1000.0
  if records:
    return json_ok({
      "found": True,
      "decision": "semantic",
      "dataset_id": dataset_id,
      "file_hash": file_hash,
      "took_ms": round(took_ms, 2),
      "records": records,
    })

  # 3) None – client may offer to escalate to LLM via a separate endpoint
  return json_ok({
    "found": False,
    "decision": "none",
    "dataset_id": dataset_id,
    "file_hash": file_hash,
    "took_ms": round(took_ms, 2),
    "records": [],
  })