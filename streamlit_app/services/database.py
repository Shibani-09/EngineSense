import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

HISTORY_REPORT_LIMIT_BYTES = 256 * 1024


def get_setting(name: str, default: str | None = None) -> str | None:
    """Read an environment value first, then the matching Streamlit secret."""

    value = os.getenv(name)
    if value is not None:
        return value

    try:
        import streamlit as st
        from streamlit.errors import StreamlitSecretNotFoundError
    except ImportError:
        return default

    try:
        secret = st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        secret = None
    return str(secret) if secret is not None else default


def is_auth_enabled() -> bool:
    """Return whether protected login is explicitly enabled."""

    value = get_setting("AUTH_ENABLED", "false")
    normalized = (value or "false").strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError("AUTH_ENABLED must be a boolean value.")


def is_mongodb_configured() -> bool:
    return bool(get_setting("MONGODB_URI") and get_setting("MONGODB_DATABASE"))


@lru_cache(maxsize=1)
def get_mongo_client() -> MongoClient[dict[str, Any]]:
    """Create one pooled MongoDB client after checking server connectivity."""

    mongo_uri = get_setting("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI is not configured.")

    client: MongoClient[dict[str, Any]] = MongoClient(
        mongo_uri,
        appname="predictive-maintenance-streamlit",
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=10000,
        maxPoolSize=10,
        minPoolSize=0,
    )
    try:
        client.admin.command("ping")
    except PyMongoError:
        client.close()
        raise
    return client


def get_database():
    database_name = get_setting("MONGODB_DATABASE")
    if not database_name:
        raise ValueError("MONGODB_DATABASE is not configured.")
    return get_mongo_client()[database_name]


def get_users_collection():
    return get_database()["users"]


def get_prediction_history_collection():
    return get_database()["prediction_history"]


def save_prediction_history(
    user_id: str,
    source_name: str,
    prediction_timestamp: str,
    record_count: int,
    equipment_count: int,
    average_predicted_rul: float,
    health_distribution: dict[str, int],
    report_csv: str | None = None,
) -> None:
    """Store per-user prediction metadata and only small downloadable reports."""

    if not user_id:
        raise ValueError("A signed-in user is required to save prediction history.")
    if record_count < 0 or equipment_count < 0:
        raise ValueError("Prediction counts cannot be negative.")
    if report_csv is not None:
        report_size = len(report_csv.encode("utf-8"))
        if report_size > HISTORY_REPORT_LIMIT_BYTES:
            report_csv = None

    document: dict[str, Any] = {
        "user_id": user_id,
        "source_name": source_name,
        "prediction_timestamp": prediction_timestamp,
        "record_count": record_count,
        "equipment_count": equipment_count,
        "average_predicted_rul": average_predicted_rul,
        "health_distribution": health_distribution,
    }
    if report_csv is not None:
        document["report_csv"] = report_csv
    get_prediction_history_collection().insert_one(document)


def get_prediction_history(
    user_id: str,
    limit: int = 50,
    include_reports: bool = True,
) -> list[dict[str, Any]]:
    """Fetch only one user's recent prediction metadata and bounded reports."""

    if not user_id:
        raise ValueError("A signed-in user is required to view prediction history.")
    if not 1 <= limit <= 100:
        raise ValueError("History limit must be between 1 and 100.")

    projection = {
        "source_name": 1,
        "prediction_timestamp": 1,
        "record_count": 1,
        "equipment_count": 1,
        "average_predicted_rul": 1,
        "health_distribution": 1,
    }
    if include_reports:
        projection["report_csv"] = 1
    documents = (
        get_prediction_history_collection()
        .find({"user_id": user_id}, projection)
        .sort("prediction_timestamp", -1)
        .limit(limit)
    )

    history = []
    for document in documents:
        document["id"] = str(document.pop("_id"))
        history.append(document)
    return history
