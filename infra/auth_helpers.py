from __future__ import annotations
from functools import wraps
from typing import Callable, Optional

from flask import session, redirect, url_for, request, flash, current_app

# Optional SQL support if available in the project
try:
    from sqlalchemy import text  # type: ignore
    _HAS_SA = True
except Exception:  # pragma: no cover
    _HAS_SA = False

# Try to load a project-local engine provider lazily.
# Adjust the import below if your project exposes it differently.
_DEF_ENGINE_FN_NAMES = (
    "get_engine",
    "get_db_engine",
    "engine",
)

def _get_engine_from_app():
    """Try to get SQLAlchemy engine from Flask app config."""
    try:
        eng = current_app.config.get("DB_ENGINE")
        return eng
    except Exception:
        return None

def _get_engine_optional():
    """Attempt to obtain a SQLAlchemy engine from infra.*
    Returns engine or None if not available.
    """
    if not _HAS_SA:
        return None
    # Try common locations
    for mod_name in ("infra.db", "infra.database", "infra.sql", "infra.repo"):
        try:
            mod = __import__(mod_name, fromlist=["*"])  # type: ignore
        except Exception:
            continue
        for fn in _DEF_ENGINE_FN_NAMES:
            eng = getattr(mod, fn, None)
            if eng is None:
                continue
            # If it's a callable, call it; otherwise assume it's already an engine
            try:
                return eng() if callable(eng) else eng
            except Exception:
                continue
    return None


def get_current_user_id() -> Optional[int]:
    """Return the current logged-in user id from session, if any."""
    uid = session.get("user_id")
    try:
        return int(uid) if uid is not None else None
    except Exception:
        return None


def _has_consent_session() -> bool:
    """Fast-path: check consent flag cached in session."""
    if session.get("user_consent", False):
        return True
    # Timestamp variant (e.g., consent_given_at stored as string)
    return bool(session.get("consent_given_at"))


def _has_consent_db(user_id: int) -> bool:
    """Slow-path: lookup consent in DB. Falls back to False if no engine available."""
    if not _HAS_SA:
        return False
    engine = _get_engine_from_app() or _get_engine_optional()
    if engine is None:
        return False
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT consent_given_at FROM users WHERE id = :uid"),
                {"uid": user_id},
            ).fetchone()
        if not row:
            return False
        consent_given_at = row[0]
        if consent_given_at:
            # Cache in session to avoid repeated DB hits
            session["consent_given_at"] = str(consent_given_at)
            session["user_consent"] = True
            return True
        return False
    except Exception:
        # Be conservative on errors: treat as no consent
        return False


def has_user_consent(user_id: Optional[int]) -> bool:
    """Public helper to check if current user has consent.
    Checks session first, then DB.
    """
    if not user_id:
        return False
    if _has_consent_session():
        return True
    return _has_consent_db(user_id)


def login_required(view: Callable):
    """Decorator: require a logged-in user. Redirects to /auth/login with next=..."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        uid = get_current_user_id()
        if uid is None:
            flash("Bitte melde dich zuerst an.", "warning")
            next_url = request.full_path if request.query_string else request.path
            return redirect(url_for("auth.login", next=next_url))
        return view(*args, **kwargs)
    return wrapped


def consent_required(view: Callable):
    """Decorator: require that the logged-in user has granted consent.
    If not logged in → behaves like login_required. If logged in but without consent → redirect to /auth/consent.
    """
    @wraps(view)
    def wrapped(*args, **kwargs):
        uid = get_current_user_id()
        if uid is None:
            flash("Bitte melde dich zuerst an.", "warning")
            next_url = request.full_path if request.query_string else request.path
            return redirect(url_for("auth.login", next=next_url))
        if not has_user_consent(uid):
            flash("Bitte erteile zuerst dein Einverständnis zur Datennutzung.", "info")
            next_url = request.full_path if request.query_string else request.path
            return redirect(url_for("auth.consent", next=next_url))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view: Callable):
    """Decorator: require that the logged-in user is an admin.
    If not logged in → behaves like login_required.
    If logged in but not admin → redirect to /auth/login with message.
    """
    @wraps(view)
    def wrapped(*args, **kwargs):
        uid = get_current_user_id()
        if uid is None:
            flash("Bitte melde dich zuerst an.", "warning")
            next_url = request.full_path if request.query_string else request.path
            return redirect(url_for("auth.login", next=next_url))
        engine = _get_engine_from_app() or _get_engine_optional()
        if engine is None:
            flash("Keine Datenbankverbindung für Admin-Prüfung.", "danger")
            return redirect(url_for("auth.login"))
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT is_admin FROM users WHERE id = :uid"),
                    {"uid": uid},
                ).fetchone()
            if not row or not bool(row[0]):
                flash("Admin-Berechtigung erforderlich.", "danger")
                return redirect(url_for("auth.login"))
        except Exception:
            flash("Fehler bei der Admin-Prüfung.", "danger")
            return redirect(url_for("auth.login"))
        return view(*args, **kwargs)
    return wrapped


def set_session_consent(consented: bool = True):
    """Helper to cache consent in the session after POST /auth/consent."""
    if consented:
        session["user_consent"] = True
        # Optionally set a simple timestamp string marker; the route can store the real DB value
        session.setdefault("consent_given_at", "set")
    else:
        session.pop("user_consent", None)
        session.pop("consent_given_at", None)