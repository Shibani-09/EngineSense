
import os
from pathlib import Path

from dotenv import load_dotenv

# Project root is the parent folder of src
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load the project's .env file
load_dotenv(PROJECT_ROOT / ".env")


def get_path(env_name: str, default: str) -> Path:
    """Read a path from .env and return it as an absolute Path."""
    value = os.getenv(env_name, default)
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


# Dataset paths
TRAIN_DATA_PATH = get_path(
    "TRAIN_DATA_PATH", "data/raw/train_FD001.txt"
)
TEST_DATA_PATH = get_path(
    "TEST_DATA_PATH", "data/raw/test_FD001.txt"
)
RUL_DATA_PATH = get_path(
    "RUL_DATA_PATH", "data/raw/RUL_FD001.txt"
)

PROCESSED_TRAIN_PATH = get_path(
    "PROCESSED_TRAIN_PATH", "data/processed/train_features.csv"
)
PROCESSED_TEST_PATH = get_path(
    "PROCESSED_TEST_PATH", "data/processed/test_features.csv"
)

# Output directories
MODEL_DIR = get_path("MODEL_DIR", "models")
OUTPUT_DIR = get_path("OUTPUT_DIR", "outputs")
PREDICTION_DIR = get_path(
    "PREDICTION_DIR", "outputs/predictions"
)
POWERBI_DIR = get_path("POWERBI_DIR", "powerbi")

# Saved model files
RF_MODEL_PATH = get_path(
    "RF_MODEL_PATH", "models/random_forest_rul_model.pkl"
)
FEATURE_COLUMNS_PATH = get_path(
    "FEATURE_COLUMNS_PATH", "models/feature_columns.pkl"
)