import pandas as pd
from pathlib import Path


# ============================================================
# PREDICTIVE MAINTENANCE PROJECT
# NASA C-MAPSS FD001 DATA LOADER
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

# Find the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Location of the raw NASA dataset
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# Dataset files
TRAIN_FILE = RAW_DATA_DIR / "train_FD001.txt"
TEST_FILE = RAW_DATA_DIR / "test_FD001.txt"
RUL_FILE = RAW_DATA_DIR / "RUL_FD001.txt"


# ------------------------------------------------------------
# 2. COLUMN NAMES
# ------------------------------------------------------------

COLUMNS = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]


# ------------------------------------------------------------
# 3. LOAD TRAINING DATA
# ------------------------------------------------------------

def load_train_data():

    df = pd.read_csv(
        TRAIN_FILE,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        engine="python"
    )

    return df


# ------------------------------------------------------------
# 4. LOAD TEST DATA
# ------------------------------------------------------------

def load_test_data():

    df = pd.read_csv(
        TEST_FILE,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        engine="python"
    )

    return df


# ------------------------------------------------------------
# 5. LOAD RUL DATA
# ------------------------------------------------------------

def load_rul_data():

    rul = pd.read_csv(
        RUL_FILE,
        sep=r"\s+",
        header=None,
        names=["RUL"],
        engine="python"
    )

    return rul


# ------------------------------------------------------------
# 6. MAIN PROGRAM
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("PREDICTIVE MAINTENANCE PROJECT")
    print("NASA C-MAPSS FD001 DATA LOADING")
    print("=" * 60)

    # Load datasets
    train_df = load_train_data()
    test_df = load_test_data()
    rul_df = load_rul_data()

    # --------------------------------------------------------
    # TRAIN DATA
    # --------------------------------------------------------

    print("\nTRAINING DATA")
    print("-" * 60)

    print("Rows:", train_df.shape[0])
    print("Columns:", train_df.shape[1])

    print("\nFirst 5 rows:")
    print(train_df.head())

    # --------------------------------------------------------
    # TEST DATA
    # --------------------------------------------------------

    print("\nTEST DATA")
    print("-" * 60)

    print("Rows:", test_df.shape[0])
    print("Columns:", test_df.shape[1])

    print("\nFirst 5 rows:")
    print(test_df.head())

    # --------------------------------------------------------
    # RUL DATA
    # --------------------------------------------------------

    print("\nRUL DATA")
    print("-" * 60)

    print("Rows:", rul_df.shape[0])
    print("Columns:", rul_df.shape[1])

    print("\nFirst 5 RUL values:")
    print(rul_df.head())

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    print("\nDATASET INFORMATION")
    print("-" * 60)

    print("Number of training engines:",
          train_df["engine_id"].nunique())

    print("Number of test engines:",
          test_df["engine_id"].nunique())

    print("Number of sensor columns:",
          len([col for col in COLUMNS if col.startswith("sensor_")]))

    print("\n" + "=" * 60)
    print("DATA LOADING COMPLETED SUCCESSFULLY")
    print("=" * 60)