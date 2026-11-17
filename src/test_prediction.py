import os
import pandas as pd
import joblib


# ============================================================
# PREDICTIVE MAINTENANCE - OFFICIAL TEST PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("PREDICTIVE MAINTENANCE - OFFICIAL TEST PREDICTION")
print("=" * 60)


# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

TEST_FILE = "data/processed/test_features.csv"
MODEL_FILE = "models/random_forest_rul_model.pkl"
FEATURE_FILE = "models/feature_columns.pkl"
OUTPUT_FILE = "outputs/predictions/test_predictions.csv"


# ------------------------------------------------------------
# 2. Load processed test data
# ------------------------------------------------------------

print("\n1. Loading processed test data...")

df = pd.read_csv(TEST_FILE)

print(f"   Test rows: {len(df)}")
print(f"   Test columns: {len(df.columns)}")


# ------------------------------------------------------------
# 3. Load trained model
# ------------------------------------------------------------

print("\n2. Loading trained Random Forest model...")

model = joblib.load(MODEL_FILE)

print("   ✓ Model loaded successfully")


# ------------------------------------------------------------
# 4. Load feature list
# ------------------------------------------------------------

print("\n3. Loading feature list...")

features = joblib.load(FEATURE_FILE)

print(f"   Number of features: {len(features)}")


# ------------------------------------------------------------
# 5. Prepare test features
# ------------------------------------------------------------

print("\n4. Preparing test features...")

X_test = df[features]

print(f"   Feature matrix shape: {X_test.shape}")


# ------------------------------------------------------------
# 6. Generate predictions
# ------------------------------------------------------------

print("\n5. Generating RUL predictions...")

predictions = model.predict(X_test)

print("   ✓ Predictions generated")


# ------------------------------------------------------------
# 7. Create prediction dataframe
# ------------------------------------------------------------

prediction_df = pd.DataFrame({
    "engine_id": df["engine_id"],
    "cycle": df["cycle"],
    "predicted_RUL": predictions
})


# ------------------------------------------------------------
# 8. Get final prediction for each engine
# ------------------------------------------------------------

final_predictions = (
    prediction_df
    .sort_values(["engine_id", "cycle"])
    .groupby("engine_id")
    .tail(1)
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 9. Create output directory
# ------------------------------------------------------------

os.makedirs("outputs/predictions", exist_ok=True)


# ------------------------------------------------------------
# 10. Save all predictions
# ------------------------------------------------------------

prediction_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n6. Saving predictions...")
print(f"   ✓ All predictions saved to:")
print(f"   {OUTPUT_FILE}")


# ------------------------------------------------------------
# 11. Display final engine predictions
# ------------------------------------------------------------

print("\n7. FINAL RUL PREDICTIONS")

print("-" * 60)

print(final_predictions.head(10).to_string(index=False))


# ------------------------------------------------------------
# 12. Prediction statistics
# ------------------------------------------------------------

print("\n8. PREDICTION STATISTICS")

print("-" * 60)

print(
    f"Minimum predicted RUL: "
    f"{predictions.min():.2f}"
)

print(
    f"Maximum predicted RUL: "
    f"{predictions.max():.2f}"
)

print(
    f"Average predicted RUL: "
    f"{predictions.mean():.2f}"
)


print("\n" + "=" * 60)
print("OFFICIAL TEST PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nNext step:")
print("Create final prediction analysis and dashboard data.")