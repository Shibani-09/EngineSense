
from io import BytesIO
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import COLUMNS
from src.feature_engineering import add_change_features, add_rolling_features


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIRECTORY = PROJECT_ROOT / "data"

MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_rul_model.pkl"

FEATURE_COLUMNS_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"

PROCESSED_TEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test_features.csv"
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_model():
    """Load the existing Random Forest model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    try:
        return joblib.load(MODEL_PATH)

    except Exception as exc:
        raise RuntimeError(
            f"Could not load Random Forest model: {exc}"
        ) from exc


# --------------------------------------------------
# LOAD FEATURE COLUMNS
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_feature_columns():
    """Load the original model feature columns."""

    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Feature columns file not found: {FEATURE_COLUMNS_PATH}"
        )

    try:
        columns = joblib.load(FEATURE_COLUMNS_PATH)

        if hasattr(columns, "tolist"):
            columns = columns.tolist()

        return list(columns)

    except Exception as exc:
        raise RuntimeError(
            f"Could not load feature columns: {exc}"
        ) from exc


# --------------------------------------------------
# LOAD PROCESSED TEST DATASET
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_test_dataset():
    """Load the processed NASA test dataset."""

    if not PROCESSED_TEST_PATH.exists():
        raise FileNotFoundError(
            f"Processed test dataset not found: {PROCESSED_TEST_PATH}"
        )

    try:
        df = pd.read_csv(PROCESSED_TEST_PATH)

    except Exception as exc:
        raise RuntimeError(
            f"Could not read test dataset: {exc}"
        ) from exc

    required_columns = {"engine_id", "cycle"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain engine_id and cycle columns."
        )

    return df


def list_existing_datasets() -> list[str]:
    """Return CSV datasets under data/, using paths relative to that directory."""

    if not DATA_DIRECTORY.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIRECTORY}")

    return sorted(
        path.relative_to(DATA_DIRECTORY).as_posix()
        for path in DATA_DIRECTORY.rglob("*.csv")
        if path.is_file()
    )


def load_existing_dataset(dataset_name: str) -> pd.DataFrame:
    """Load a project CSV without allowing paths outside data/."""

    data_root = DATA_DIRECTORY.resolve()
    path = (data_root / dataset_name).resolve()
    if path.suffix.lower() != ".csv" or data_root not in path.parents:
        raise ValueError("Select a CSV dataset inside the project data directory.")
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_name}")

    try:
        dataframe = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read dataset {dataset_name}: {exc}") from exc

    return dataframe


def validate_processed_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate processed model inputs without imputing or fabricating values."""

    if dataframe.empty:
        raise ValueError("The dataset contains no records.")

    required_columns = list(
        dict.fromkeys(["engine_id", "cycle", *load_feature_columns()])
    )
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            "The processed dataset is missing required columns: "
            + ", ".join(missing_columns)
            + "."
        )

    validated = dataframe.copy()
    invalid_columns = []
    for column in dict.fromkeys(required_columns):
        converted = pd.to_numeric(validated[column], errors="coerce")
        numeric_values = converted.to_numpy(dtype=float, na_value=np.nan)
        invalid_mask = converted.isna() | ~np.isfinite(numeric_values)
        if invalid_mask.any():
            invalid_columns.append(f"{column} ({int(invalid_mask.sum())} invalid)")
        else:
            validated[column] = converted

    if invalid_columns:
        raise ValueError(
            "Required dataset columns must contain finite numeric values: "
            + ", ".join(invalid_columns)
            + "."
        )

    for column in ("engine_id", "cycle"):
        values = validated[column].to_numpy(dtype=float)
        if ((values <= 0) | (values % 1 != 0)).any():
            raise ValueError(
                f"The {column} column must contain positive whole numbers."
            )

    return validated


def predict_existing_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Predict every processed record, preserving all input columns."""

    validated = validate_processed_dataset(dataframe)
    if "predicted_rul" in validated.columns:
        raise ValueError(
            "The input already contains a predicted_rul column; rename it "
            "before running inference to preserve the original records."
        )
    feature_columns = load_feature_columns()
    model = load_model()
    if getattr(model, "n_features_in_", len(feature_columns)) != len(feature_columns):
        raise ValueError(
            "The saved model input count does not match feature_columns.pkl."
        )

    predictions = np.asarray(
        model.predict(validated.loc[:, feature_columns]), dtype=float
    )

    if predictions.shape != (len(validated),) or not np.isfinite(predictions).all():
        raise ValueError("The model returned invalid RUL predictions.")

    results = validated.copy()
    results["predicted_rul"] = np.maximum(0.0, predictions)
    return results


# --------------------------------------------------
# PREDICT RUL
# --------------------------------------------------

def predict_rul(features: dict) -> float:
    """Predict remaining useful life from prepared features."""

    model = load_model()
    feature_columns = load_feature_columns()

    missing_features = [
        column
        for column in feature_columns
        if column not in features
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    try:
        input_data = pd.DataFrame(
            [[features[column] for column in feature_columns]],
            columns=feature_columns
        )

        prediction = model.predict(input_data)[0]

        return max(0.0, float(prediction))

    except Exception as exc:
        raise RuntimeError(
            f"RUL prediction failed: {exc}"
        ) from exc


# --------------------------------------------------
# PREDICT FOR SELECTED ENGINE
# --------------------------------------------------

def predict_engine_rul(engine_id: int) -> dict:
    """Predict RUL using the latest processed row of an engine."""

    df = load_test_dataset()

    feature_columns = load_feature_columns()

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing model features: {missing_columns}"
        )

    engine_rows = df[df["engine_id"] == engine_id]

    if engine_rows.empty:
        raise ValueError(
            f"Engine ID {engine_id} was not found."
        )

    latest_row = engine_rows.sort_values(
        "cycle"
    ).iloc[-1]

    features = {
        column: float(latest_row[column])
        for column in feature_columns
    }

    predicted_rul = predict_rul(features)

    return {
        "engine_id": int(engine_id),
        "cycle": int(latest_row["cycle"]),
        "predicted_rul": round(predicted_rul, 2),
        "unit": "cycles"
    }


# --------------------------------------------------
# GET AVAILABLE ENGINE IDS
# --------------------------------------------------

def get_available_engines() -> list:
    """Return all available engine IDs."""

    df = load_test_dataset()

    return sorted(
        df["engine_id"].dropna().astype(int).unique().tolist()
    )


# --------------------------------------------------
# UPLOADED RAW SENSOR CSV PREDICTIONS
# --------------------------------------------------

RAW_SENSOR_COLUMNS = [
    column for column in COLUMNS if column.startswith("sensor_")
]
MIN_HISTORY_CYCLES = 5


def load_uploaded_sensor_csv(filename: str, content: bytes) -> pd.DataFrame:
    """Read an uploaded CSV containing raw C-MAPSS sensor records."""

    if Path(filename).suffix.lower() != ".csv":
        raise ValueError("Unsupported file format. Please upload a .csv file.")

    if not content:
        raise ValueError("The uploaded CSV file is empty.")

    try:
        return pd.read_csv(BytesIO(content))
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded CSV is empty or has no header.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"The uploaded CSV could not be parsed: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError("The uploaded CSV must use UTF-8 text encoding.") from exc
    except OSError as exc:
        raise ValueError(f"The uploaded CSV could not be read: {exc}") from exc


def validate_uploaded_sensor_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize raw C-MAPSS rows before feature engineering."""

    if dataframe.empty:
        raise ValueError("The CSV contains no data rows.")

    def record_numbers(mask: pd.Series | np.ndarray) -> str:
        positions = np.flatnonzero(np.asarray(mask, dtype=bool)) + 1
        shown = ", ".join(str(position) for position in positions[:20])
        if len(positions) > 20:
            shown += f", and {len(positions) - 20} more"
        return shown

    expected_columns = list(COLUMNS)
    missing_columns = [
        column for column in expected_columns if column not in dataframe.columns
    ]
    unexpected_columns = [
        str(column)
        for column in dataframe.columns
        if column not in expected_columns
    ]
    if missing_columns or unexpected_columns:
        details = []
        if missing_columns:
            details.append("missing columns: " + ", ".join(missing_columns))
        if unexpected_columns:
            details.append("unexpected columns: " + ", ".join(unexpected_columns))
        raise ValueError(
            "The CSV must use the raw C-MAPSS schema (" + "; ".join(details) + ")."
        )

    validated = dataframe.copy()
    missing_values = validated.isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        details = ", ".join(
            f"{column} ({count})"
            for column, count in missing_values.items()
        )
        missing_records = validated.loc[:, missing_values.index].isna().any(axis=1)
        raise ValueError(
            f"The CSV contains missing values: {details}. Invalid record numbers "
            f"(1-based): {record_numbers(missing_records)}."
        )

    invalid_values = []
    invalid_records = pd.Series(False, index=validated.index)
    for column in expected_columns:
        converted = pd.to_numeric(validated[column], errors="coerce")
        invalid_mask = converted.isna() & validated[column].notna()
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            invalid_values.append(f"{column} ({invalid_count})")
            invalid_records |= invalid_mask
        validated[column] = converted

    if invalid_values:
        raise ValueError(
            "All C-MAPSS columns must contain numeric values; invalid data in: "
            + ", ".join(invalid_values)
            + f". Invalid record numbers (1-based): {record_numbers(invalid_records)}."
        )

    finite_values = np.isfinite(validated.to_numpy(dtype=float))
    if not finite_values.all():
        invalid_records = ~finite_values.all(axis=1)
        invalid_columns = [
            column
            for column in expected_columns
            if not np.isfinite(validated[column].to_numpy(dtype=float)).all()
        ]
        raise ValueError(
            "The CSV contains non-finite numeric values in: "
            + ", ".join(invalid_columns)
            + f". Invalid record numbers (1-based): {record_numbers(invalid_records)}."
            + "."
        )

    for column in ("engine_id", "cycle"):
        values = validated[column].to_numpy(dtype=float)
        invalid_records = (values <= 0) | (values % 1 != 0)
        if invalid_records.any():
            raise ValueError(
                f"The {column} column must contain positive whole numbers. "
                f"Invalid record numbers (1-based): "
                f"{record_numbers(invalid_records)}."
            )

    duplicate_cycles = validated.duplicated(
        subset=["engine_id", "cycle"], keep=False
    )
    if duplicate_cycles.any():
        raise ValueError(
            "The CSV contains duplicate engine_id/cycle records at record "
            f"numbers (1-based): {record_numbers(duplicate_cycles)}."
        )

    insufficient_history = []
    for engine_id, engine_rows in validated.groupby("engine_id", sort=False):
        cycles = np.sort(engine_rows["cycle"].to_numpy(dtype=int))
        if len(cycles) < MIN_HISTORY_CYCLES:
            insufficient_history.append(str(int(engine_id)))
        elif not np.all(np.diff(cycles[-MIN_HISTORY_CYCLES:]) == 1):
            insufficient_history.append(str(int(engine_id)))

    if insufficient_history:
        raise ValueError(
            "Each engine needs at least five observations with consecutive latest "
            "cycles for the rolling features. Check engine IDs: "
            + ", ".join(insufficient_history)
            + "."
        )

    return validated.reset_index(drop=True)


def predict_uploaded_engines(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Predict every raw CSV record and preserve its original input columns."""

    sensor_data = validate_uploaded_sensor_data(dataframe)
    feature_columns = load_feature_columns()

    if len(feature_columns) != 64 or len(set(feature_columns)) != 64:
        raise ValueError(
            "The saved feature list must contain exactly 64 unique model features."
        )

    active_sensors = [
        column
        for column in RAW_SENSOR_COLUMNS
        if column in feature_columns
    ]
    expected_features = (
        ["cycle", "setting_1", "setting_2", "setting_3"]
        + active_sensors
        + [
            feature
            for sensor in active_sensors
            for feature in (
                f"{sensor}_rolling_mean",
                f"{sensor}_rolling_std",
            )
        ]
        + [f"{sensor}_change" for sensor in active_sensors]
    )
    if feature_columns != expected_features:
        raise ValueError(
            "The saved feature list does not match the existing C-MAPSS "
            "feature-engineering pipeline."
        )

    engineered = add_rolling_features(sensor_data, active_sensors)
    engineered = add_change_features(engineered, active_sensors)

    missing_features = [
        column for column in feature_columns if column not in engineered.columns
    ]
    if missing_features:
        raise ValueError(
            "Feature engineering did not produce required model features: "
            + ", ".join(missing_features)
            + "."
        )

    predictions = np.asarray(
        load_model().predict(engineered.loc[:, feature_columns]),
        dtype=float,
    )

    if predictions.shape != (len(sensor_data),) or not np.isfinite(predictions).all():
        raise ValueError("The model returned invalid RUL predictions.")

    prediction_rows = engineered.loc[:, ["engine_id", "cycle"]].copy()
    prediction_rows["predicted_rul"] = np.maximum(0.0, predictions)
    return sensor_data.merge(
        prediction_rows,
        on=["engine_id", "cycle"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
