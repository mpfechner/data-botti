from werkzeug.security import check_password_hash
from datetime import datetime
from infra.auth_helpers import get_current_user_id, set_session_consent
from flask import Blueprint, render_template, request, redirect, url_for, session, current_app, flash
from sqlalchemy import text

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        engine = current_app.config.get("DB_ENGINE")
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT id, username, password_hash, is_admin FROM users WHERE email = :email"),
                {"email": email},
            ).fetchone()

        if row and check_password_hash(row[2], password):
            session['user_id'] = row[0]
            session['username'] = row[1]
            session['is_admin'] = bool(row[3])
            # Clear any previous login-required flashes set by guards
            session.pop('_flashes', None)
            # Prefer ?next=... but also accept hidden form field if present
            next_url = request.args.get('next') or request.form.get('next') or url_for('app_dashboard')
            return redirect(next_url)
        else:
            return render_template('login.html', error="Ungültige Anmeldedaten")
    return render_template('login.html')

@auth_bp.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('index'))

@auth_bp.route('/consent', methods=['GET', 'POST'])
def consent():
    uid = get_current_user_id()
    if not uid:
        return redirect(url_for('auth.login', next=request.path))

    if request.method == 'POST':
        engine = current_app.config.get("DB_ENGINE")
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE users SET consent_given_at = :now WHERE id = :uid"),
                {"now": datetime.utcnow(), "uid": uid},
            )
        set_session_consent(True)
        # Clear any previous warning/info flashes (e.g., from the guard) and show success
        session.pop('_flashes', None)
        flash("Danke, dein Einverständnis wurde gespeichert.", "success")
        next_url = request.args.get('next') or url_for('app_dashboard')
        return redirect(next_url)

    return render_template('consent.html')