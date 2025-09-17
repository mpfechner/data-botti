from __future__ import annotations
from flask import Blueprint, render_template, request, redirect, url_for, flash
from werkzeug.security import generate_password_hash

from infra.auth_helpers import login_required, admin_required, get_current_user_id

# Repo functions (live in repo.py at project root)
from repo import (
    repo_list_users,
    repo_create_user,
    repo_update_user_password,
    repo_delete_user,
    repo_list_groups,
    repo_create_group,
    repo_delete_group,
    repo_list_group_members,
    repo_add_user_to_group,
    repo_remove_user_from_group,
    repo_is_user_admin,
    repo_count_admins,
    DuplicateEmailError,
    DuplicateGroupNameError,
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
    except DuplicateEmailError:
        flash("Diese E-Mail-Adresse existiert bereits.", "warning")
    except Exception as e:
        msg = str(e) or "Unbekannter Fehler beim Anlegen."
        flash(f"Fehler beim Anlegen: {msg}", "danger")
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
        # Guard 1: no self-deletion
        current_id = get_current_user_id()
        if int(user_id) == int(current_id):
            flash("Du kannst deinen eigenen Account nicht löschen.", "warning")
            return redirect(url_for("admin.list_users"))
        # Guard 2: do not delete the last admin
        if repo_is_user_admin(user_id=user_id):
            if repo_count_admins() <= 1:
                flash("Der letzte Admin darf nicht gelöscht werden.", "warning")
                return redirect(url_for("admin.list_users"))
        # proceed with deletion
        repo_delete_user(user_id=user_id)
        flash("Nutzer gelöscht.", "success")
    except Exception as e:
        flash(f"Fehler beim Löschen: {e}", "danger")
    return redirect(url_for("admin.list_users"))


# -------------------- Groups --------------------
@bp.route("/groups", methods=["GET"])
@login_required
@admin_required
def list_groups():
    groups = repo_list_groups()
    users = repo_list_users()
    # Build members-by-group map
    members_by_group = {}
    for g in groups:
        members_by_group[g["id"]] = repo_list_group_members(group_id=g["id"])  # list of mappings
    return render_template("admin_groups.html", groups=groups, users=users, members=members_by_group)


@bp.route("/groups/create", methods=["POST"])
@login_required
@admin_required
def create_group():
    name = request.form.get("name", "").strip()
    if not name:
        flash("Gruppenname ist erforderlich.", "warning")
        return redirect(url_for("admin.list_groups"))
    try:
        repo_create_group(name=name)
        flash("Gruppe angelegt.", "success")
    except DuplicateGroupNameError:
        flash("Diese Gruppe existiert bereits.", "warning")
    except Exception as e:
        msg = str(e) or "Unbekannter Fehler beim Anlegen der Gruppe."
        flash(f"Fehler beim Anlegen der Gruppe: {msg}", "danger")
    return redirect(url_for("admin.list_groups"))


@bp.route("/groups/<int:group_id>/delete", methods=["POST"])
@login_required
@admin_required
def delete_group(group_id: int):
    try:
        repo_delete_group(group_id=group_id)
        flash("Gruppe gelöscht.", "success")
    except Exception as e:
        flash(f"Fehler beim Löschen der Gruppe: {e}", "danger")
    return redirect(url_for("admin.list_groups"))


@bp.route("/groups/<int:group_id>/add_user", methods=["POST"])
@login_required
@admin_required
def add_user_to_group(group_id: int):
    user_id = request.form.get("user_id")
    if not user_id:
        flash("User-ID fehlt.", "warning")
        return redirect(url_for("admin.list_groups"))
    try:
        repo_add_user_to_group(user_id=int(user_id), group_id=group_id)
        flash("Nutzer zur Gruppe hinzugefügt.", "success")
    except Exception as e:
        flash(f"Fehler beim Hinzufügen: {e}", "danger")
    return redirect(url_for("admin.list_groups"))


@bp.route("/groups/<int:group_id>/remove_user", methods=["POST"])
@login_required
@admin_required
def remove_user_from_group(group_id: int):
    user_id = request.form.get("user_id")
    if not user_id:
        flash("User-ID fehlt.", "warning")
        return redirect(url_for("admin.list_groups"))
    try:
        repo_remove_user_from_group(user_id=int(user_id), group_id=group_id)
        flash("Nutzer aus Gruppe entfernt.", "success")
    except Exception as e:
        flash(f"Fehler beim Entfernen: {e}", "danger")
    return redirect(url_for("admin.list_groups"))