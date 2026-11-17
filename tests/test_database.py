from streamlit_app.services import database


class MemoryCursor:
    def sort(self, *args):
        del args
        return self

    def limit(self, limit):
        del limit
        return self

    def __iter__(self):
        return iter([])


class MemoryHistory:
    def __init__(self):
        self.inserted = []
        self.query = None

    def insert_one(self, document):
        self.inserted.append(document)

    def find(self, query, projection):
        del projection
        self.query = query
        return MemoryCursor()


def test_authentication_is_disabled_unless_explicitly_enabled(monkeypatch):
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    assert not database.is_auth_enabled()
    monkeypatch.setenv("AUTH_ENABLED", "true")
    assert database.is_auth_enabled()
    monkeypatch.setenv("AUTH_ENABLED", "invalid")
    try:
        database.is_auth_enabled()
    except ValueError as error:
        assert "AUTH_ENABLED" in str(error)
    else:
        raise AssertionError("Invalid AUTH_ENABLED value was accepted.")


def test_history_is_filtered_by_the_signed_in_user(monkeypatch):
    collection = MemoryHistory()
    monkeypatch.setattr(
        database, "get_prediction_history_collection", lambda: collection
    )

    database.save_prediction_history(
        user_id="account-a",
        source_name="processed/test_features.csv",
        prediction_timestamp="2026-01-01T00:00:00+00:00",
        record_count=10,
        equipment_count=2,
        average_predicted_rul=20.0,
        health_distribution={"Critical": 2},
        report_csv="small report",
    )
    database.get_prediction_history("account-b", include_reports=False)

    assert collection.inserted[0]["user_id"] == "account-a"
    assert collection.inserted[0]["report_csv"] == "small report"
    assert collection.query == {"user_id": "account-b"}


def test_oversized_csv_report_is_not_stored(monkeypatch):
    collection = MemoryHistory()
    monkeypatch.setattr(
        database, "get_prediction_history_collection", lambda: collection
    )
    database.save_prediction_history(
        user_id="account-a",
        source_name="large-upload.csv",
        prediction_timestamp="2026-01-01T00:00:00+00:00",
        record_count=100000,
        equipment_count=100,
        average_predicted_rul=55.0,
        health_distribution={"Healthy": 100},
        report_csv="x" * (database.HISTORY_REPORT_LIMIT_BYTES + 1),
    )

    assert "report_csv" not in collection.inserted[0]
