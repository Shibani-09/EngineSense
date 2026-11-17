import pandas as pd
from pathlib import Path


# ============================================================
# PREDICTIVE MAINTENANCE PROJECT
# DATA QUALITY ANALYSIS
# NASA C-MAPSS FD001
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT PATH
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"

TRAIN_FILE = RAW_DATA_DIR / "train_FD001.txt"


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
# 3. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(
    TRAIN_FILE,
    sep=r"\s+",
    header=None,
    names=COLUMNS,
    engine="python"
)


print("=" * 70)
print("PREDICTIVE MAINTENANCE - DATA QUALITY ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# 4. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("\n1. DATASET INFORMATION")
print("-" * 70)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("Number of engines:",
      df["engine_id"].nunique())


# ------------------------------------------------------------
# 5. MISSING VALUES
# ------------------------------------------------------------

print("\n2. MISSING VALUE CHECK")
print("-" * 70)

missing_values = df.isnull().sum()

total_missing = missing_values.sum()

print("Total missing values:", total_missing)

if total_missing == 0:
    print("Status: No missing values found.")
else:
    print("\nColumns with missing values:")
    print(missing_values[missing_values > 0])


# ------------------------------------------------------------
# 6. DUPLICATE ROWS
# ------------------------------------------------------------

print("\n3. DUPLICATE ROW CHECK")
print("-" * 70)

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)

if duplicate_count == 0:
    print("Status: No duplicate rows found.")
else:
    print("Warning: Duplicate rows detected.")


# ------------------------------------------------------------
# 7. SENSOR COLUMNS
# ------------------------------------------------------------

sensor_columns = [
    column for column in df.columns
    if column.startswith("sensor_")
]


print("\n4. SENSOR INFORMATION")
print("-" * 70)

print("Number of sensors:", len(sensor_columns))

print("\nSensor columns:")

for sensor in sensor_columns:
    print(sensor)


# ------------------------------------------------------------
# 8. CONSTANT SENSORS
# ------------------------------------------------------------

print("\n5. CONSTANT SENSOR CHECK")
print("-" * 70)

constant_sensors = []

for sensor in sensor_columns:

    unique_values = df[sensor].nunique()

    if unique_values <= 1:
        constant_sensors.append(sensor)


if constant_sensors:

    print("Constant sensors found:")

    for sensor in constant_sensors:
        print(sensor)

else:

    print("No completely constant sensors found.")


# ------------------------------------------------------------
# 9. SENSOR STATISTICS
# ------------------------------------------------------------

print("\n6. SENSOR STATISTICS")
print("-" * 70)

sensor_statistics = df[sensor_columns].describe().T

print(
    sensor_statistics[
        ["mean", "std", "min", "max"]
    ].round(3)
)


# ------------------------------------------------------------
# 10. SENSOR VARIABILITY
# ------------------------------------------------------------

print("\n7. SENSOR VARIABILITY")
print("-" * 70)

sensor_variability = df[sensor_columns].std().sort_values()

print("Sensors with lowest variability:")

print(sensor_variability.head(10).round(4))


print("\nSensors with highest variability:")

print(sensor_variability.tail(10).round(4))


# ------------------------------------------------------------
# 11. ENGINE LIFETIME
# ------------------------------------------------------------

print("\n8. ENGINE LIFETIME ANALYSIS")
print("-" * 70)

engine_lifetime = (
    df.groupby("engine_id")["cycle"]
    .max()
)


print("Minimum engine lifetime:",
      engine_lifetime.min(), "cycles")

print("Maximum engine lifetime:",
      engine_lifetime.max(), "cycles")

print("Average engine lifetime:",
      round(engine_lifetime.mean(), 2), "cycles")


# ------------------------------------------------------------
# 12. FIRST AND LAST ENGINE
# ------------------------------------------------------------

print("\n9. SAMPLE ENGINE LIFETIMES")
print("-" * 70)

print(
    engine_lifetime.head(10)
)


# ------------------------------------------------------------
# 13. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA QUALITY ANALYSIS COMPLETED")
print("=" * 70)

print("\nSummary:")

print("✓ Dataset loaded successfully")

print("✓ Missing values checked")

print("✓ Duplicate rows checked")

print("✓ Constant sensors checked")

print("✓ Sensor statistics calculated")

print("✓ Sensor variability calculated")

print("✓ Engine lifetimes calculated")

print("\nNext step:")
print("Feature engineering and RUL calculation")