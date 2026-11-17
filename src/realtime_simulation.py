import time
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PREDICTIVE MAINTENANCE - REAL-TIME STREAMING SIMULATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DATA = BASE_DIR / "data" / "processed" / "test_features.csv"
MODEL_FILE = BASE_DIR / "models" / "random_forest_rul_model.pkl"
FEATURE_FILE = BASE_DIR / "models" / "feature_columns.pkl"

OUTPUT_DIR = BASE_DIR / "outputs" / "predictions"
OUTPUT_FILE = OUTPUT_DIR / "realtime_predictions.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 65)
print("PREDICTIVE MAINTENANCE - REAL-TIME STREAMING SIMULATION")
print("=" * 65)


# ------------------------------------------------------------
# 1. Load processed test data
# ------------------------------------------------------------

print("\n1. Loading processed test data...")

df = pd.read_csv(TEST_DATA)

print(f"   Total records: {len(df)}")
print(f"   Total engines: {df['engine_id'].nunique()}")


# ------------------------------------------------------------
# 2. Load trained model
# ------------------------------------------------------------

print("\n2. Loading trained Random Forest model...")

model = joblib.load(MODEL_FILE)

print("   ✓ Model loaded successfully")


# ------------------------------------------------------------
# 3. Load feature list
# ------------------------------------------------------------

print("\n3. Loading feature list...")

feature_columns = joblib.load(FEATURE_FILE)

print(f"   Features required: {len(feature_columns)}")


# ------------------------------------------------------------
# 4. Prepare features
# ------------------------------------------------------------

X = df[feature_columns].copy()

print("\n4. Feature matrix prepared...")
print(f"   Shape: {X.shape}")


# ------------------------------------------------------------
# 5. Maintenance status
# ------------------------------------------------------------

def get_status(rul):

    if rul <= 30:
        return "CRITICAL"

    elif rul <= 60:
        return "WARNING"

    elif rul <= 100:
        return "MONITOR"

    else:
        return "HEALTHY"


# ------------------------------------------------------------
# 6. Risk score
# ------------------------------------------------------------

def get_risk_score(rul):

    if rul <= 30:
        return 4

    elif rul <= 60:
        return 3

    elif rul <= 100:
        return 2

    else:
        return 1


# ------------------------------------------------------------
# 7. Maintenance recommendation
# ------------------------------------------------------------

def get_recommendation(status):

    if status == "CRITICAL":
        return "Immediate Maintenance Required"

    elif status == "WARNING":
        return "Schedule Maintenance Soon"

    elif status == "MONITOR":
        return "Monitor Equipment"

    else:
        return "No Immediate Action Required"


# ------------------------------------------------------------
# 8. Start streaming
# ------------------------------------------------------------

print("\n5. Starting real-time sensor stream...")
print("   One record will arrive every second.")
print("   Press CTRL+C to stop.")

print("\n" + "=" * 65)


results = []


try:

    for index in range(len(df)):

        row = df.iloc[index]

        features = X.iloc[[index]]

        # Predict RUL
        predicted_rul = float(model.predict(features)[0])

        predicted_rul = max(0, predicted_rul)

        # Status
        status = get_status(predicted_rul)

        # Risk
        risk_score = get_risk_score(predicted_rul)

        # Recommendation
        recommendation = get_recommendation(status)

        # Current timestamp
        timestamp = pd.Timestamp.now()

        # Save result
        result = {
            "timestamp": timestamp,
            "engine_id": int(row["engine_id"]),
            "cycle": int(row["cycle"]),
            "predicted_RUL": round(predicted_rul, 2),
            "risk_score": risk_score,
            "maintenance_status": status,
            "maintenance_recommendation": recommendation
        }

        results.append(result)

        # Display live result
        print(
            f"\nTimestamp      : {timestamp}"
            f"\nEngine ID      : {int(row['engine_id'])}"
            f"\nCycle          : {int(row['cycle'])}"
            f"\nPredicted RUL  : {predicted_rul:.2f} cycles"
            f"\nRisk Score     : {risk_score}/4"
            f"\nStatus         : {status}"
            f"\nRecommendation : {recommendation}"
        )

        print("-" * 65)

        # Save continuously
        pd.DataFrame(results).to_csv(
            OUTPUT_FILE,
            index=False
        )

        # Simulate incoming sensor data
        time.sleep(1)


except KeyboardInterrupt:

    print("\n\nStreaming simulation stopped by user.")


finally:

    # Save all records collected before stopping
    pd.DataFrame(results).to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nReal-time results saved to:")
    print(f"   {OUTPUT_FILE}")


print("\n" + "=" * 65)
print("REAL-TIME STREAMING SIMULATION FINISHED")
print("=" * 65)