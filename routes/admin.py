

from __future__ import annotations
from flask import Blueprint, render_template, request, redirect, url_for, flash
from werkzeug.security import generate_password_hash

from infra.auth_helpers import login_required, admin_required

# Repo functions (live in repo.py at project root)
from repo import (
    repo_list_users,
    repo_create_user,
    repo_update_user_password,
    repo_delete_user,
)

bp = Blueprint("admin", __name__, url_prefix="/admin")


@bp.route("/users", methods=["GET"])
@login_required
@admin_required
def list_users():
    users = repo_list_users()
    return render_template("admin_users.html", users=users)


@bp.route("/users/create", methods=["POST"])
@login_required
@admin_required
def create_user():
    email = request.form.get("email", "").strip()
    username = request.form.get("username", "").strip() or None
    password = request.form.get("password", "")
    is_admin = 1 if request.form.get("is_admin") else 0

    if not email or not password:
        flash("E-Mail und Passwort sind erforderlich.", "warning")
        return redirect(url_for("admin.list_users"))

    pw_hash = generate_password_hash(password)

    try:
        repo_create_user(email=email, username=username, password_hash=pw_hash, is_admin=is_admin)
        flash("Nutzer wurde angelegt.", "success")
    except Exception as e:
        flash(f"Fehler beim Anlegen: {e}", "danger")
    return redirect(url_for("admin.list_users"))


@bp.route("/users/<int:user_id>/password", methods=["POST"])
@login_required
@admin_required
def change_password(user_id: int):
    new_password = request.form.get("password", "")
    if not new_password:
        flash("Neues Passwort fehlt.", "warning")
        return redirect(url_for("admin.list_users"))

    pw_hash = generate_password_hash(new_password)

    try:
        repo_update_user_password(user_id=user_id, password_hash=pw_hash)
        flash("Passwort aktualisiert.", "success")
    except Exception as e:
        flash(f"Fehler beim Aktualisieren des Passworts: {e}", "danger")
    return redirect(url_for("admin.list_users"))


@bp.route("/users/<int:user_id>/delete", methods=["POST"])
@login_required
@admin_required
def delete_user(user_id: int):
    try:
        repo_delete_user(user_id=user_id)
        flash("Nutzer gelöscht.", "success")
    except Exception as e:
        flash(f"Fehler beim Löschen: {e}", "danger")
    return redirect(url_for("admin.list_users"))