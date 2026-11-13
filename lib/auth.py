"""Authentication and authorization service for SecondSelf.

Provides secure password hashing (PBKDF2-HMAC-SHA256 with 260,000 iterations),
user registration, authentication, session lifecycle management, password reset flows,
and server-side verified administrative controls.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import logging
import os
from pathlib import Path
import re
import secrets
from typing import Any

try:
    from lib.database import (
        DEFAULT_ADMIN_EMAIL,
        DEFAULT_USER_ID,
        create_password_reset,
        create_session,
        create_user,
        delete_session,
        get_password_reset,
        get_session,
        get_user_by_email,
        get_user_by_id,
        list_all_users,
        mark_password_reset_used,
        update_user,
    )
    from lib.models import User, UserSession
except ImportError:
    from .database import (
        DEFAULT_ADMIN_EMAIL,
        DEFAULT_USER_ID,
        create_password_reset,
        create_session,
        create_user,
        delete_session,
        get_password_reset,
        get_session,
        get_user_by_email,
        get_user_by_id,
        list_all_users,
        mark_password_reset_used,
        update_user,
    )
    from .models import User, UserSession

logger = logging.getLogger(__name__)

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8
DEFAULT_SESSION_DAYS = 7


def hash_password(password: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with 260,000 rounds and random 16-byte salt."""
    if not isinstance(password, str) or not password:
        raise ValueError("Password must be a non-empty string.")
    salt = secrets.token_bytes(16)
    iterations = 260_000
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${derived.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify plain password against PBKDF2 formatted hash using constant-time comparison."""
    if not isinstance(password, str) or not isinstance(hashed, str):
        return False
    try:
        parts = hashed.split("$")
        if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
            return False
        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        expected_derived = bytes.fromhex(parts[3])
        computed_derived = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations
        )
        return secrets.compare_digest(expected_derived, computed_derived)
    except Exception:
        return False


def validate_email(email: str) -> bool:
    """Return True if email matches basic format."""
    if not email or not isinstance(email, str):
        return False
    return bool(EMAIL_REGEX.match(email.strip()))


def register_user(
    email: str,
    password: str,
    full_name: str = "",
    role: str = "user",
    root: Path | str | None = None,
) -> tuple[bool, str, User | None]:
    """Register a new user account.

    If registering the default admin account (admin@secondself.ai) and it already exists
    with placeholder password, updates the existing admin user with the supplied password.
    """
    clean_email = email.strip().lower() if email else ""
    if not validate_email(clean_email):
        return False, "Invalid email address format.", None

    if len(password) < MIN_PASSWORD_LENGTH:
        return (
            False,
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.",
            None,
        )

    existing = get_user_by_email(clean_email, root=root)
    hashed = hash_password(password)

    if existing is not None:
        user, old_hash = existing
        # If this is the pre-seeded default admin account and the password was placeholder
        if user.id == DEFAULT_USER_ID and clean_email == DEFAULT_ADMIN_EMAIL:
            update_user(
                user.id,
                {
                    "password_hash": hashed,
                    "full_name": full_name or user.full_name or "SecondSelf Admin",
                    "email_verified": True,
                    "is_active": True,
                },
                root=root,
            )
            updated_user = get_user_by_id(user.id, root=root)
            return True, "Default administrator account activated.", updated_user
        return False, "An account with this email address already exists.", None

    # Check if there are no existing custom users in the system; if so, make first user admin
    all_users = list_all_users(root=root)
    # Filter out initial unactivated placeholder if any
    custom_users = [u for u in all_users if u["id"] != DEFAULT_USER_ID]
    user_role = "admin" if (len(custom_users) == 0 and role == "admin") or (clean_email == DEFAULT_ADMIN_EMAIL) else role

    new_user = create_user(
        email=clean_email,
        password_hash=hashed,
        full_name=full_name.strip(),
        role=user_role,
        is_active=True,
        email_verified=True,
        root=root,
    )
    return True, "Account registered successfully.", new_user


def login_user(
    email: str,
    password: str,
    root: Path | str | None = None,
) -> tuple[bool, str, UserSession | None, User | None]:
    """Authenticate credentials and issue a new secure session token."""
    clean_email = email.strip().lower() if email else ""
    if not clean_email or not password:
        return False, "Please enter both email and password.", None, None

    result = get_user_by_email(clean_email, root=root)
    if result is None:
        return False, "Invalid email or password.", None, None

    user, stored_hash = result

    if not verify_password(password, stored_hash):
        return False, "Invalid email or password.", None, None

    if not user.is_active:
        return (
            False,
            "Your account is inactive or disabled. Please contact the administrator.",
            None,
            None,
        )

    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=DEFAULT_SESSION_DAYS)
    token = secrets.token_urlsafe(32)

    create_session(
        token=token,
        user_id=user.id,
        expires_at=expires.isoformat(),
        root=root,
    )

    update_user(user.id, {"last_login_at": now.isoformat()}, root=root)
    refreshed_user = get_user_by_id(user.id, root=root) or user

    session = UserSession(
        token=token,
        user_id=user.id,
        expires_at=expires.isoformat(),
        created_at=now.isoformat(),
    )

    return True, "Login successful.", session, refreshed_user


def authenticate_session(
    token: str | None,
    root: Path | str | None = None,
) -> User | None:
    """Validate a session token and return the active User, or None if invalid/expired."""
    if not token or not isinstance(token, str):
        return None
    return get_session(token.strip(), root=root)


def logout_user(
    token: str | None,
    root: Path | str | None = None,
) -> bool:
    """Invalidate and remove the active session token."""
    if not token or not isinstance(token, str):
        return False
    delete_session(token.strip(), root=root)
    return True


def request_password_reset(
    email: str,
    root: Path | str | None = None,
) -> tuple[bool, str, str | None]:
    """Generate a password reset token for an account."""
    clean_email = email.strip().lower() if email else ""
    if not validate_email(clean_email):
        return False, "Please enter a valid email address.", None

    result = get_user_by_email(clean_email, root=root)
    if result is None:
        # Generic message to prevent email enumeration, but return False
        return (
            True,
            "If that email exists in SecondSelf, reset instructions have been generated.",
            None,
        )

    user, _ = result
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    create_password_reset(token, user.id, expires_at, root=root)

    return True, "Password reset token generated.", token


def reset_password(
    token: str,
    new_password: str,
    root: Path | str | None = None,
) -> tuple[bool, str]:
    """Reset a user's password using a valid reset token."""
    if not token or not isinstance(token, str):
        return False, "Invalid or missing reset token."

    if len(new_password) < MIN_PASSWORD_LENGTH:
        return (
            False,
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.",
        )

    record = get_password_reset(token.strip(), root=root)
    if not record:
        return False, "Reset token is invalid or has expired."

    user_id = record["user_id"]
    new_hash = hash_password(new_password)

    update_user(user_id, {"password_hash": new_hash}, root=root)
    mark_password_reset_used(token.strip(), root=root)

    return True, "Password successfully reset. You can now log in."


def verify_admin(
    actor: User | None,
    root: Path | str | None = None,
) -> bool:
    """Server-side verification of administrator status directly from database."""
    if not actor or not actor.id:
        return False
    db_user = get_user_by_id(actor.id, root=root)
    if not db_user or not db_user.is_active:
        return False
    return db_user.role == "admin"


def admin_list_users(
    actor: User | None,
    root: Path | str | None = None,
) -> list[dict[str, Any]]:
    """List all registered users with metrics for administrator view."""
    if not verify_admin(actor, root=root):
        raise PermissionError("Access denied: Administrator privileges required.")
    return list_all_users(root=root)


def admin_set_user_status(
    actor: User | None,
    target_user_id: str,
    is_active: bool,
    root: Path | str | None = None,
) -> tuple[bool, str]:
    """Activate or deactivate a user account."""
    if not verify_admin(actor, root=root):
        raise PermissionError("Access denied: Administrator privileges required.")

    if actor and actor.id == target_user_id and not is_active:
        return False, "Administrators cannot deactivate their own account."

    target_user = get_user_by_id(target_user_id, root=root)
    if not target_user:
        return False, "Target user not found."

    ok = update_user(target_user_id, {"is_active": is_active}, root=root)
    status_str = "activated" if is_active else "deactivated"
    return ok, f"User {target_user.email} has been {status_str}."


def admin_set_user_role(
    actor: User | None,
    target_user_id: str,
    new_role: str,
    root: Path | str | None = None,
) -> tuple[bool, str]:
    """Assign a role ('admin' or 'user') to a target user."""
    if not verify_admin(actor, root=root):
        raise PermissionError("Access denied: Administrator privileges required.")

    if new_role not in {"admin", "user"}:
        return False, "Role must be 'admin' or 'user'."

    if actor and actor.id == target_user_id and new_role != "admin":
        return False, "Administrators cannot revoke their own admin role."

    target_user = get_user_by_id(target_user_id, root=root)
    if not target_user:
        return False, "Target user not found."

    ok = update_user(target_user_id, {"role": new_role}, root=root)
    return ok, f"User {target_user.email} role updated to {new_role}."
