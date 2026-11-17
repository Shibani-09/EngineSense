import re
from datetime import datetime, timezone

import bcrypt
from pymongo.errors import DuplicateKeyError

from streamlit_app.services.database import get_users_collection


_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MAX_BCRYPT_PASSWORD_BYTES = 72


def _validate_registration(full_name: str, email: str, password: str) -> str | None:
    if not full_name.strip() or not email.strip() or not password:
        return "All fields are required."
    if len(full_name.strip()) > 120:
        return "Full name must be 120 characters or fewer."
    normalized_email = email.strip().lower()
    if len(normalized_email) > 254 or not _EMAIL_PATTERN.fullmatch(normalized_email):
        return "Enter a valid email address."
    if len(password) < 8:
        return "Password must contain at least 8 characters."
    if len(password.encode("utf-8")) > _MAX_BCRYPT_PASSWORD_BYTES:
        return "Password must be no more than 72 UTF-8 bytes."
    return None


def hash_password(password: str) -> str:
    """Hash a password with bcrypt; stored hashes remain Passlib-compatible."""

    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(),
    ).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a bcrypt hash, returning false for malformed stored hashes."""

    if len(password.encode("utf-8")) > _MAX_BCRYPT_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("ascii"),
        )
    except (UnicodeEncodeError, ValueError, TypeError):
        return False


def register_user(full_name: str, email: str, password: str) -> tuple[bool, str]:
    """Create an account without storing a plaintext password."""

    error = _validate_registration(full_name, email, password)
    if error:
        return False, error

    collection = get_users_collection()
    normalized_email = email.strip().lower()
    if collection.find_one({"normalized_email": normalized_email}):
        return False, "An account with this email already exists."

    user = {
        "full_name": full_name.strip(),
        "email": normalized_email,
        "normalized_email": normalized_email,
        "password_hash": hash_password(password),
        "failed_login_attempts": 0,
        "is_locked": False,
        "created_at": datetime.now(timezone.utc),
    }

    try:
        collection.insert_one(user)
    except DuplicateKeyError as error:
        key_pattern = (error.details or {}).get("keyPattern", {})
        if "normalized_email" in key_pattern:
            return False, "An account with this email already exists."
        raise

    return True, "Account created successfully."


def authenticate_user(
    email: str,
    password: str,
) -> tuple[bool, str, dict[str, str] | None]:
    """Verify account credentials and return only non-secret identity fields."""

    normalized_email = email.strip().lower()
    if not normalized_email or not password:
        return False, "Invalid email or password.", None

    collection = get_users_collection()
    user = collection.find_one({"normalized_email": normalized_email})
    if not user:
        return False, "Invalid email or password.", None

    if user.get("is_locked", False):
        return False, "Your account is locked. Contact the administrator.", None

    if not verify_password(password, user.get("password_hash", "")):
        attempts = int(user.get("failed_login_attempts", 0)) + 1
        collection.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "failed_login_attempts": attempts,
                    "is_locked": attempts >= 3,
                }
            },
        )
        if attempts >= 3:
            return False, "Account locked after 3 failed attempts.", None
        return False, f"Invalid email or password. {3 - attempts} attempts remaining.", None

    collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"failed_login_attempts": 0}},
    )
    return True, "Login successful.", {
        "id": str(user["_id"]),
        "full_name": user.get("full_name", ""),
        "email": user.get("email", normalized_email),
    }
