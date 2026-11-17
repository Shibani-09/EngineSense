import bcrypt

from streamlit_app.services import authentication


class MemoryUsers:
    def __init__(self):
        self.documents = []

    def find_one(self, query):
        return next(
            (
                document
                for document in self.documents
                if document.get("normalized_email") == query["normalized_email"]
            ),
            None,
        )

    def insert_one(self, document):
        self.documents.append(document.copy())

    def update_one(self, query, update):
        document = next(
            item for item in self.documents if item["_id"] == query["_id"]
        )
        document.update(update["$set"])


def test_registration_normalizes_email_and_stores_only_bcrypt_hash(monkeypatch):
    collection = MemoryUsers()
    monkeypatch.setattr(authentication, "get_users_collection", lambda: collection)

    success, message = authentication.register_user(
        "  Ada Engineer  ",
        "ADA@example.com",
        "safe-password-123",
    )

    assert success
    assert message == "Account created successfully."
    user = collection.documents[0]
    assert user["normalized_email"] == "ada@example.com"
    assert user["full_name"] == "Ada Engineer"
    assert "password" not in user
    assert bcrypt.checkpw(
        b"safe-password-123",
        user["password_hash"].encode("ascii"),
    )


def test_registration_rejects_invalid_email_and_duplicate(monkeypatch):
    collection = MemoryUsers()
    monkeypatch.setattr(authentication, "get_users_collection", lambda: collection)
    assert authentication.register_user(
        "Ada", "not-an-email", "safe-password-123"
    ) == (False, "Enter a valid email address.")
    assert collection.documents == []

    collection.documents.append(
        {
            "normalized_email": "ada@example.com",
            "password_hash": "not-exposed",
        }
    )
    success, message = authentication.register_user(
        "Ada", "ADA@example.com", "safe-password-123"
    )
    assert not success
    assert "already exists" in message

    too_long_password = "é" * 37
    assert authentication.register_user(
        "Ada", "another@example.com", too_long_password
    ) == (False, "Password must be no more than 72 UTF-8 bytes.")


def test_login_tracks_failures_locks_account_and_returns_safe_identity(monkeypatch):
    collection = MemoryUsers()
    collection.documents.append(
        {
            "_id": "user-1",
            "full_name": "Ada",
            "email": "ada@example.com",
            "normalized_email": "ada@example.com",
            "password_hash": authentication.hash_password("correct-password"),
            "failed_login_attempts": 0,
            "is_locked": False,
        }
    )
    monkeypatch.setattr(authentication, "get_users_collection", lambda: collection)

    assert not authentication.authenticate_user(
        "ada@example.com", "incorrect-password"
    )[0]
    assert not authentication.authenticate_user(
        "ada@example.com", "incorrect-password"
    )[0]
    locked = authentication.authenticate_user(
        "ada@example.com", "incorrect-password"
    )
    assert locked == (
        False,
        "Account locked after 3 failed attempts.",
        None,
    )

    collection.documents[0]["is_locked"] = False
    success, _, user = authentication.authenticate_user(
        "ADA@example.com", "correct-password"
    )
    assert success
    assert user == {
        "id": "user-1",
        "full_name": "Ada",
        "email": "ada@example.com",
    }
    assert "password_hash" not in user
