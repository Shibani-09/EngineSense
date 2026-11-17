import pandas as pd
import pytest

from streamlit_app.services.ml_service import (
    list_existing_datasets,
    load_existing_dataset,
    predict_existing_dataset,
    predict_uploaded_engines,
    validate_uploaded_sensor_data,
)
from src.data_loader import COLUMNS


def test_existing_dataset_prediction_preserves_records_and_columns():
    dataset_name = "processed/test_features.csv"
    original = load_existing_dataset(dataset_name)
    results = predict_existing_dataset(original)

    assert len(results) == len(original)
    assert list(results.columns[: len(original.columns)]) == list(original.columns)
    assert "predicted_rul" in results.columns
    assert results["predicted_rul"].ge(0).all()


def test_dataset_catalog_is_limited_to_project_data():
    datasets = list_existing_datasets()

    assert "processed/test_features.csv" in datasets
    assert all(name.lower().endswith(".csv") for name in datasets)
    with pytest.raises(ValueError, match="inside the project data directory"):
        load_existing_dataset("../.env")


def test_uploaded_raw_csv_predicts_all_original_records():
    training_data = pd.read_csv(
        "data/raw/train_FD001.txt",
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )
    uploaded_data = training_data.loc[
        (training_data["engine_id"] == 1) & (training_data["cycle"] <= 8)
    ].sample(frac=1, random_state=42)

    validated = validate_uploaded_sensor_data(uploaded_data)
    results = predict_uploaded_engines(validated)

    assert len(results) == len(uploaded_data)
    assert list(results.columns[: len(COLUMNS)]) == COLUMNS
    pd.testing.assert_frame_equal(
        results[["engine_id", "cycle"]].reset_index(drop=True),
        uploaded_data[["engine_id", "cycle"]].reset_index(drop=True),
    )
    assert results["predicted_rul"].ge(0).all()


def test_uploaded_csv_rejects_missing_sensor_columns():
    invalid_data = pd.DataFrame(
        [{"engine_id": 1, "cycle": 1, "setting_1": 0.0}]
    )

    with pytest.raises(ValueError, match="missing columns"):
        validate_uploaded_sensor_data(invalid_data)


def test_uploaded_csv_reports_invalid_record_numbers():
    training_data = pd.read_csv(
        "data/raw/train_FD001.txt",
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )
    uploaded_data = training_data.loc[
        (training_data["engine_id"] == 1) & (training_data["cycle"] <= 5)
    ].copy()
    uploaded_data.loc[uploaded_data.index[1], "sensor_2"] = float("nan")

    with pytest.raises(ValueError, match=r"record numbers \(1-based\): 2"):
        validate_uploaded_sensor_data(uploaded_data)
