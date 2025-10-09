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
def get_body_field(field_name: str, default: Any = None):
    data_json = request.get_json(silent=True) or {}
    data_form = request.form or {}
    value = data_json.get(field_name) or data_form.get(field_name)
    if isinstance(value, str):
        value = value.strip()
    return value if value is not None else default

@api_v1.post("/search")
@api_auth_required
def api_search_only():
    from services.search_service import SearchService
    user_id = current_api_user_id()
    prompt = get_body_field("prompt")
    if not prompt:
        prompt = get_body_field("q")
    dataset_id = int(get_body_field("dataset_id"))
    top_k = int(get_body_field("top_k") or 3)
    # Optional: include_seeds (default True). Accepts bool, 0/1, "true"/"false", etc.
    raw_include = get_body_field("include_seeds", True)
    include_seeds = True
    if isinstance(raw_include, bool):
        include_seeds = raw_include
    elif isinstance(raw_include, str):
        include_seeds = raw_include.strip().lower() in ("1", "true", "yes", "y", "on")
    elif isinstance(raw_include, (int, float)):
        try:
            include_seeds = int(raw_include) != 0
        except Exception:
            include_seeds = True
    # Frontend safeguard: if request comes from browser (no API token), never include seeds
    if getattr(g, "api_token", None) is None:
        include_seeds = False
    # --- logging: request summary ---
    t0 = perf_counter()
    try:
        q_preview = (prompt or "")[:80].replace("\n", " ")
        q_len = len(prompt or "")
    except Exception:
        q_preview, q_len = "", 0
    current_app.logger.info(
        "api/search start uid=%s dataset_id=%s top_k=%s q='%s' len=%s include_seeds=%s",
        user_id, dataset_id, top_k, q_preview, q_len, include_seeds
    )
    result = SearchService.suggest_similar_questions(prompt, dataset_id, user_id, top_k=top_k, include_seeds=include_seeds)
    # --- logging: result summary ---
    try:
        elapsed_ms = (perf_counter() - t0) * 1000.0
        if isinstance(result, dict):
            decision = result.get("decision")
            found = result.get("found")
            records = result.get("records") or []
            n = len(records)
            top_score = None
            if records and isinstance(records[0], dict):
                ts = records[0].get("score")
                if isinstance(ts, (int, float)):
                    top_score = f"{ts:.4f}"
            current_app.logger.info(
                "api/search done decision=%s found=%s n=%s top_score=%s dt_ms=%.1f",
                decision, found, n, top_score, elapsed_ms
            )
        else:
            current_app.logger.info(
                "api/search done (non-dict result) dt_ms=%.1f",
                elapsed_ms
            )
    except Exception:
        current_app.logger.exception("api/search logging failed")
    return json_ok(result)