import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PREDICTIVE MAINTENANCE PROJECT
# FEATURE ENGINEERING + RUL CALCULATION
# NASA C-MAPSS FD001
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

TRAIN_FILE = RAW_DATA_DIR / "train_FD001.txt"
TEST_FILE = RAW_DATA_DIR / "test_FD001.txt"
RUL_FILE = RAW_DATA_DIR / "RUL_FD001.txt"


# Create processed-data folder if it doesn't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


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
# 3. SENSOR COLUMNS
# ------------------------------------------------------------

SENSOR_COLUMNS = [
    column
    for column in COLUMNS
    if column.startswith("sensor_")
]


# ------------------------------------------------------------
# 4. LOAD DATA
# ------------------------------------------------------------

def load_data(file_path):

    return pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        engine="python"
    )


# ------------------------------------------------------------
# 5. CALCULATE TRAINING RUL
# ------------------------------------------------------------

def calculate_training_rul(df):

    # Find the final cycle for every engine
    max_cycles = (
        df.groupby("engine_id")["cycle"]
        .max()
        .reset_index()
    )

    max_cycles.rename(
        columns={"cycle": "max_cycle"},
        inplace=True
    )

    # Add maximum cycle to every row
    df = df.merge(
        max_cycles,
        on="engine_id",
        how="left"
    )

    # RUL = final cycle - current cycle
    df["RUL"] = (
        df["max_cycle"] - df["cycle"]
    )

    # Remove helper column
    df.drop(
        columns=["max_cycle"],
        inplace=True
    )

    return df


# ------------------------------------------------------------
# 6. CALCULATE TEST RUL
# ------------------------------------------------------------

def calculate_test_rul(test_df, rul_df):

    # Find the last observed cycle in test data
    max_cycles = (
        test_df.groupby("engine_id")["cycle"]
        .max()
        .reset_index()
    )

    max_cycles.rename(
        columns={"cycle": "max_test_cycle"},
        inplace=True
    )

    # RUL file contains the remaining life AFTER
    # the final observed test cycle.
    rul_df = rul_df.copy()

    rul_df["engine_id"] = (
        np.arange(1, len(rul_df) + 1)
    )

    rul_df.rename(
        columns={"RUL": "final_RUL"},
        inplace=True
    )

    # Combine test cycle information with official RUL
    test_df = test_df.merge(
        max_cycles,
        on="engine_id",
        how="left"
    )

    test_df = test_df.merge(
        rul_df,
        on="engine_id",
        how="left"
    )

    # Calculate RUL for every test observation
    test_df["RUL"] = (
        test_df["final_RUL"]
        + test_df["max_test_cycle"]
        - test_df["cycle"]
    )

    # Remove helper columns
    test_df.drop(
        columns=[
            "max_test_cycle",
            "final_RUL"
        ],
        inplace=True
    )

    return test_df


# ------------------------------------------------------------
# 7. REMOVE CONSTANT SENSORS
# ------------------------------------------------------------

def remove_constant_sensors(train_df, test_df):

    constant_sensors = []

    for sensor in SENSOR_COLUMNS:

        if train_df[sensor].nunique() <= 1:
            constant_sensors.append(sensor)

    train_df = train_df.drop(
        columns=constant_sensors
    )

    test_df = test_df.drop(
        columns=constant_sensors
    )

    return train_df, test_df, constant_sensors


# ------------------------------------------------------------
# 8. ADD ROLLING FEATURES
# ------------------------------------------------------------

def add_rolling_features(df, sensor_columns):

    df = df.copy()

    # Sort chronologically
    df.sort_values(
        ["engine_id", "cycle"],
        inplace=True
    )

    # Rolling window size
    window = 5

    for sensor in sensor_columns:

        # Rolling mean
        df[f"{sensor}_rolling_mean"] = (
            df.groupby("engine_id")[sensor]
            .transform(
                lambda x:
                x.rolling(
                    window=window,
                    min_periods=1
                ).mean()
            )
        )

        # Rolling standard deviation
        df[f"{sensor}_rolling_std"] = (
            df.groupby("engine_id")[sensor]
            .transform(
                lambda x:
                x.rolling(
                    window=window,
                    min_periods=1
                ).std()
            )
        )

    # Replace initial NaN values created by rolling std
    df.fillna(0, inplace=True)

    return df


# ------------------------------------------------------------
# 9. ADD SENSOR CHANGE FEATURES
# ------------------------------------------------------------

def add_change_features(df, sensor_columns):

    df = df.copy()

    df.sort_values(
        ["engine_id", "cycle"],
        inplace=True
    )

    for sensor in sensor_columns:

        # Difference from previous cycle
        df[f"{sensor}_change"] = (
            df.groupby("engine_id")[sensor]
            .diff()
            .fillna(0)
        )

    return df


# ------------------------------------------------------------
# 10. MAIN PROGRAM
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("PREDICTIVE MAINTENANCE")
    print("FEATURE ENGINEERING + RUL CALCULATION")
    print("=" * 70)


    # --------------------------------------------------------
    # LOAD TRAINING DATA
    # --------------------------------------------------------

    print("\n1. Loading training data...")

    train_df = load_data(TRAIN_FILE)

    print(
        "Training rows:",
        len(train_df)
    )


    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    print("\n2. Loading test data...")

    test_df = load_data(TEST_FILE)

    print(
        "Test rows:",
        len(test_df)
    )


    # --------------------------------------------------------
    # LOAD OFFICIAL RUL
    # --------------------------------------------------------

    print("\n3. Loading official test RUL...")

    rul_df = pd.read_csv(
        RUL_FILE,
        sep=r"\s+",
        header=None,
        names=["RUL"],
        engine="python"
    )

    print(
        "RUL records:",
        len(rul_df)
    )


    # --------------------------------------------------------
    # TRAINING RUL
    # --------------------------------------------------------

    print("\n4. Calculating training RUL...")

    train_df = calculate_training_rul(
        train_df
    )

    print(
        "Training RUL range:",
        train_df["RUL"].min(),
        "to",
        train_df["RUL"].max()
    )


    # --------------------------------------------------------
    # TEST RUL
    # --------------------------------------------------------

    print("\n5. Calculating test RUL...")

    test_df = calculate_test_rul(
        test_df,
        rul_df
    )

    print(
        "Test RUL range:",
        test_df["RUL"].min(),
        "to",
        test_df["RUL"].max()
    )


    # --------------------------------------------------------
    # REMOVE CONSTANT SENSORS
    # --------------------------------------------------------

    print("\n6. Removing constant sensors...")

    train_df, test_df, constant_sensors = (
        remove_constant_sensors(
            train_df,
            test_df
        )
    )

    print(
        "Removed sensors:",
        constant_sensors
    )

    print(
        "Remaining sensors:",
        len(
            [
                col
                for col in train_df.columns
                if col.startswith("sensor_")
            ]
        )
    )


    # --------------------------------------------------------
    # IDENTIFY ACTIVE SENSORS
    # --------------------------------------------------------

    active_sensors = [
        sensor
        for sensor in SENSOR_COLUMNS
        if sensor not in constant_sensors
    ]


    # --------------------------------------------------------
    # ROLLING FEATURES
    # --------------------------------------------------------

    print("\n7. Creating rolling features...")

    train_df = add_rolling_features(
        train_df,
        active_sensors
    )

    test_df = add_rolling_features(
        test_df,
        active_sensors
    )


    # --------------------------------------------------------
    # CHANGE FEATURES
    # --------------------------------------------------------

    print("\n8. Creating sensor change features...")

    train_df = add_change_features(
        train_df,
        active_sensors
    )

    test_df = add_change_features(
        test_df,
        active_sensors
    )


    # --------------------------------------------------------
    # SAVE PROCESSED DATA
    # --------------------------------------------------------

    print("\n9. Saving processed datasets...")

    train_output = (
        PROCESSED_DATA_DIR
        / "train_features.csv"
    )

    test_output = (
        PROCESSED_DATA_DIR
        / "test_features.csv"
    )

    train_df.to_csv(
        train_output,
        index=False
    )

    test_df.to_csv(
        test_output,
        index=False
    )


    # --------------------------------------------------------
    # FINAL INFORMATION
    # --------------------------------------------------------

    print("\n10. FINAL DATASET INFORMATION")

    print("-" * 70)

    print(
        "Processed training shape:",
        train_df.shape
    )

    print(
        "Processed test shape:",
        test_df.shape
    )

    print(
        "Training RUL minimum:",
        train_df["RUL"].min()
    )

    print(
        "Training RUL maximum:",
        train_df["RUL"].max()
    )

    print(
        "Processed training file:",
        train_output
    )

    print(
        "Processed test file:",
        test_output
    )


    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nNext step:")
    print("Build and evaluate the machine-learning model.")