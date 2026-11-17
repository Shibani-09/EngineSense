# ============================================================
# PREDICTIVE MAINTENANCE PROJECT
# Model Evaluation and Visualization
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PATHS
# ============================================================

DATA_FILE = "outputs/validation_predictions.csv"
MODEL_RESULTS_FILE = "outputs/model_results.csv"

CHART_DIR = "outputs/charts"

os.makedirs(CHART_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("PREDICTIVE MAINTENANCE - MODEL EVALUATION")
print("=" * 60)

print("\n1. Loading validation predictions...")

df = pd.read_csv(DATA_FILE)

print(f"   Rows: {len(df)}")
print(f"   Columns: {len(df.columns)}")


# ============================================================
# 3. CALCULATE FINAL METRICS
# ============================================================

print("\n2. Calculating evaluation metrics...")

actual = df["RUL"]
predicted = df["Predicted_RUL"]

mae = mean_absolute_error(
    actual,
    predicted
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predicted
    )
)

r2 = r2_score(
    actual,
    predicted
)

print("\n   FINAL MODEL PERFORMANCE")
print("   -----------------------")

print(f"   MAE  : {mae:.4f}")
print(f"   RMSE : {rmse:.4f}")
print(f"   R²   : {r2:.4f}")


# ============================================================
# 4. ACTUAL VS PREDICTED RUL
# ============================================================

print("\n3. Creating Actual vs Predicted RUL chart...")

plt.figure(figsize=(10, 6))

plt.scatter(
    actual,
    predicted,
    alpha=0.5
)

min_value = min(
    actual.min(),
    predicted.min()
)

max_value = max(
    actual.max(),
    predicted.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")

plt.title(
    "Actual vs Predicted Remaining Useful Life"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    f"{CHART_DIR}/actual_vs_predicted_rul.png",
    dpi=300
)

plt.close()

print("   ✓ Chart saved")


# ============================================================
# 5. PREDICTION ERROR DISTRIBUTION
# ============================================================

print("\n4. Creating prediction error chart...")

errors = actual - predicted

plt.figure(figsize=(10, 6))

plt.hist(
    errors,
    bins=40
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Prediction Error")

plt.ylabel("Frequency")

plt.title(
    "RUL Prediction Error Distribution"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    f"{CHART_DIR}/prediction_error_distribution.png",
    dpi=300
)

plt.close()

print("   ✓ Chart saved")


# ============================================================
# 6. RUL PREDICTION OVER CYCLES
# ============================================================

print("\n5. Creating RUL over cycle chart...")

# Select one validation engine
selected_engine = df["engine_id"].iloc[0]

engine_data = df[
    df["engine_id"] == selected_engine
].sort_values("cycle")


plt.figure(figsize=(12, 6))

plt.plot(
    engine_data["cycle"],
    engine_data["RUL"],
    label="Actual RUL"
)

plt.plot(
    engine_data["cycle"],
    engine_data["Predicted_RUL"],
    label="Predicted RUL"
)

plt.xlabel("Engine Cycle")

plt.ylabel("Remaining Useful Life")

plt.title(
    f"Actual vs Predicted RUL - Engine {selected_engine}"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    f"{CHART_DIR}/rul_prediction_engine_{selected_engine}.png",
    dpi=300
)

plt.close()

print("   ✓ Chart saved")


# ============================================================
# 7. MODEL COMPARISON
# ============================================================

print("\n6. Creating model comparison chart...")

results = pd.read_csv(
    MODEL_RESULTS_FILE
)

models = results["Model"]

mae_values = results["MAE"]

plt.figure(figsize=(9, 6))

plt.bar(
    models,
    mae_values
)

plt.xlabel("Model")

plt.ylabel("Mean Absolute Error")

plt.title(
    "Model Comparison - MAE"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    f"{CHART_DIR}/model_comparison_mae.png",
    dpi=300
)

plt.close()

print("   ✓ Chart saved")


# ============================================================
# 8. R² COMPARISON
# ============================================================

print("\n7. Creating R² comparison chart...")

r2_values = results["R2"]

plt.figure(figsize=(9, 6))

plt.bar(
    models,
    r2_values
)

plt.xlabel("Model")

plt.ylabel("R² Score")

plt.title(
    "Model Comparison - R² Score"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    f"{CHART_DIR}/model_comparison_r2.png",
    dpi=300
)

plt.close()

print("   ✓ Chart saved")


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

print("\n8. Creating feature importance chart...")

feature_file = "models/feature_columns.pkl"

feature_columns = joblib.load(
    feature_file
)


# Find which model was selected
best_model_name = results.loc[
    results["MAE"].idxmin(),
    "Model"
]


if best_model_name == "Random Forest":

    model_file = "models/random_forest_rul_model.pkl"

else:

    model_file = "models/xgboost_rul_model.pkl"


model = joblib.load(model_file)


if hasattr(model, "feature_importances_"):

    importance = model.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importance
    })

    feature_importance = feature_importance.sort_values(
        "Importance",
        ascending=False
    )

    top_features = feature_importance.head(15)

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1]
    )

    plt.xlabel("Importance")

    plt.ylabel("Feature")

    plt.title(
        "Top 15 Features Used for RUL Prediction"
    )

    plt.tight_layout()

    plt.savefig(
        f"{CHART_DIR}/feature_importance.png",
        dpi=300
    )

    plt.close()

    feature_importance.to_csv(
        "outputs/feature_importance.csv",
        index=False
    )

    print("   ✓ Feature importance chart saved")

else:

    print(
        "   Feature importance is not available for this model."
    )


# ============================================================
# 10. SAVE FINAL METRICS
# ============================================================

metrics = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2"
    ],
    "Value": [
        mae,
        rmse,
        r2
    ]
})

metrics.to_csv(
    "outputs/final_model_metrics.csv",
    index=False
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETED")
print("=" * 60)

print(f"\nMAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

print("\nCharts created:")

print("✓ actual_vs_predicted_rul.png")
print("✓ prediction_error_distribution.png")
print(
    f"✓ rul_prediction_engine_{selected_engine}.png"
)
print("✓ model_comparison_mae.png")
print("✓ model_comparison_r2.png")
print("✓ feature_importance.png")

print("\nFiles saved in:")
print("outputs/charts/")

print("\nNext step:")
print("Generate predictions for the official test dataset.")