# ============================================================
# PREDICTIVE MAINTENANCE PROJECT
# Machine Learning Model Training and Evaluation
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# ============================================================
# 1. FILE PATHS
# ============================================================

TRAIN_FILE = "data/processed/train_features.csv"

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD PROCESSED DATA
# ============================================================

print("=" * 60)
print("PREDICTIVE MAINTENANCE - MACHINE LEARNING")
print("=" * 60)

print("\n1. Loading processed training data...")

df = pd.read_csv(TRAIN_FILE)

print(f"   Dataset shape: {df.shape}")


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\n2. Dataset information")

print(f"   Number of rows: {len(df)}")
print(f"   Number of columns: {len(df.columns)}")
print(f"   Number of engines: {df['engine_id'].nunique()}")
print(f"   RUL minimum: {df['RUL'].min()}")
print(f"   RUL maximum: {df['RUL'].max()}")


# ============================================================
# 4. SPLIT BY ENGINE ID
# ============================================================

print("\n3. Creating engine-based train/validation split...")

unique_engines = sorted(df["engine_id"].unique())

np.random.seed(42)

np.random.shuffle(unique_engines)

split_index = int(len(unique_engines) * 0.80)

train_engines = unique_engines[:split_index]
validation_engines = unique_engines[split_index:]

train_df = df[df["engine_id"].isin(train_engines)].copy()
validation_df = df[df["engine_id"].isin(validation_engines)].copy()

print(f"   Training engines: {len(train_engines)}")
print(f"   Validation engines: {len(validation_engines)}")

print(f"   Training rows: {len(train_df)}")
print(f"   Validation rows: {len(validation_df)}")


# ============================================================
# 5. PREPARE FEATURES AND TARGET
# ============================================================

print("\n4. Preparing features...")

TARGET = "RUL"

# Columns that should not be used as ML features
DROP_COLUMNS = [
    "RUL",
    "engine_id"
]

feature_columns = [
    column
    for column in df.columns
    if column not in DROP_COLUMNS
]

X_train = train_df[feature_columns]
y_train = train_df[TARGET]

X_validation = validation_df[feature_columns]
y_validation = validation_df[TARGET]

print(f"   Number of features: {len(feature_columns)}")

print("\n   Features used:")

for feature in feature_columns:
    print(f"   - {feature}")


# ============================================================
# 6. RANDOM FOREST MODEL
# ============================================================

print("\n5. Training Random Forest model...")

random_forest = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train, y_train)

rf_predictions = random_forest.predict(X_validation)


# ============================================================
# 7. RANDOM FOREST EVALUATION
# ============================================================

rf_mae = mean_absolute_error(
    y_validation,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_validation,
    rf_predictions
)

print("\n   RANDOM FOREST RESULTS")
print("   ---------------------")
print(f"   MAE  : {rf_mae:.4f}")
print(f"   RMSE : {rf_rmse:.4f}")
print(f"   R²   : {rf_r2:.4f}")


# ============================================================
# 8. XGBOOST MODEL
# ============================================================

print("\n6. Training XGBoost model...")

xgboost_model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1
)

xgboost_model.fit(
    X_train,
    y_train
)

xgb_predictions = xgboost_model.predict(
    X_validation
)


# ============================================================
# 9. XGBOOST EVALUATION
# ============================================================

xgb_mae = mean_absolute_error(
    y_validation,
    xgb_predictions
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        xgb_predictions
    )
)

xgb_r2 = r2_score(
    y_validation,
    xgb_predictions
)

print("\n   XGBOOST RESULTS")
print("   ---------------")
print(f"   MAE  : {xgb_mae:.4f}")
print(f"   RMSE : {xgb_rmse:.4f}")
print(f"   R²   : {xgb_r2:.4f}")


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

print("\n7. MODEL COMPARISON")
print("=" * 60)

results = pd.DataFrame({
    "Model": [
        "Random Forest",
        "XGBoost"
    ],
    "MAE": [
        rf_mae,
        xgb_mae
    ],
    "RMSE": [
        rf_rmse,
        xgb_rmse
    ],
    "R2": [
        rf_r2,
        xgb_r2
    ]
})

print(results.to_string(index=False))


# ============================================================
# 11. SELECT BEST MODEL
# ============================================================

# Lower MAE is better
best_index = results["MAE"].idxmin()

best_model_name = results.loc[
    best_index,
    "Model"
]

print("\n8. BEST MODEL")
print("=" * 60)

print(f"Best model: {best_model_name}")


# ============================================================
# 12. SAVE MODEL
# ============================================================

if best_model_name == "Random Forest":

    best_model = random_forest
    model_filename = "models/random_forest_rul_model.pkl"

else:

    best_model = xgboost_model
    model_filename = "models/xgboost_rul_model.pkl"


joblib.dump(
    best_model,
    model_filename
)

print(f"Model saved to: {model_filename}")


# ============================================================
# 13. SAVE FEATURE LIST
# ============================================================

feature_file = "models/feature_columns.pkl"

joblib.dump(
    feature_columns,
    feature_file
)

print(f"Feature list saved to: {feature_file}")


# ============================================================
# 14. SAVE MODEL RESULTS
# ============================================================

results_file = "outputs/model_results.csv"

results.to_csv(
    results_file,
    index=False
)

print(f"Model results saved to: {results_file}")


# ============================================================
# 15. SAVE VALIDATION PREDICTIONS
# ============================================================

validation_results = validation_df[
    ["engine_id", "cycle", "RUL"]
].copy()

if best_model_name == "Random Forest":
    validation_results["Predicted_RUL"] = rf_predictions
else:
    validation_results["Predicted_RUL"] = xgb_predictions

validation_results["Prediction_Error"] = (
    validation_results["RUL"]
    - validation_results["Predicted_RUL"]
)

validation_file = "outputs/validation_predictions.csv"

validation_results.to_csv(
    validation_file,
    index=False
)

print(f"Validation predictions saved to: {validation_file}")


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Best Model: {best_model_name}")

print("\nFiles created:")

print(f"✓ {model_filename}")
print(f"✓ {feature_file}")
print(f"✓ {results_file}")
print(f"✓ {validation_file}")

print("\nNext step:")
print("Generate model evaluation charts and feature importance.")