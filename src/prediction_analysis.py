import os
import pandas as pd
import numpy as np


# ============================================================
# PREDICTIVE MAINTENANCE - FINAL PREDICTION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("PREDICTIVE MAINTENANCE - FINAL PREDICTION ANALYSIS")
print("=" * 60)


# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

PREDICTION_FILE = "outputs/predictions/test_predictions.csv"
OUTPUT_FILE = "outputs/predictions/final_engine_predictions.csv"


# ------------------------------------------------------------
# 2. Load predictions
# ------------------------------------------------------------

print("\n1. Loading test predictions...")

df = pd.read_csv(PREDICTION_FILE)

print(f"   Rows: {len(df)}")
print(f"   Columns: {len(df.columns)}")


# ------------------------------------------------------------
# 3. Get final prediction for each engine
# ------------------------------------------------------------

print("\n2. Calculating final RUL for each engine...")

final_df = (
    df.sort_values(["engine_id", "cycle"])
      .groupby("engine_id")
      .tail(1)
      .reset_index(drop=True)
)


# ------------------------------------------------------------
# 4. Add maintenance status
# ------------------------------------------------------------

print("\n3. Creating maintenance status...")

def maintenance_status(rul):
    if rul <= 30:
        return "Critical"
    elif rul <= 60:
        return "Warning"
    elif rul <= 120:
        return "Monitor"
    else:
        return "Healthy"


final_df["maintenance_status"] = final_df["predicted_RUL"].apply(
    maintenance_status
)


# ------------------------------------------------------------
# 5. Add risk score
# ------------------------------------------------------------

print("\n4. Creating risk score...")

def risk_score(rul):
    if rul <= 30:
        return 4
    elif rul <= 60:
        return 3
    elif rul <= 120:
        return 2
    else:
        return 1


final_df["risk_score"] = final_df["predicted_RUL"].apply(
    risk_score
)


# ------------------------------------------------------------
# 6. Sort by highest risk
# ------------------------------------------------------------

final_df = final_df.sort_values(
    ["risk_score", "predicted_RUL"],
    ascending=[False, True]
).reset_index(drop=True)



# 7. Save final engine predictions


os.makedirs("outputs/predictions", exist_ok=True)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)



# 8. Display final results


print("\n5. FINAL ENGINE PREDICTIONS")

print("-" * 60)

print(
    final_df[
        [
            "engine_id",
            "cycle",
            "predicted_RUL",
            "maintenance_status",
            "risk_score"
        ]
    ].head(20).to_string(index=False)
)



# 9. Maintenance summary


print("\n6. MAINTENANCE STATUS SUMMARY")

print("-" * 60)

status_counts = final_df["maintenance_status"].value_counts()

print(status_counts)

# 10. Critical engines

critical = final_df[
    final_df["maintenance_status"] == "Critical"
]

print("\n7. CRITICAL ENGINES")

print("-" * 60)

if len(critical) > 0:
    print(
        critical[
            [
                "engine_id",
                "cycle",
                "predicted_RUL",
                "maintenance_status"
            ]
        ].to_string(index=False)
    )
else:
    print("No critical engines found.")



# 11. Warning engines


warning = final_df[
    final_df["maintenance_status"] == "Warning"
]

print("\n8. WARNING ENGINES")

print("-" * 60)

if len(warning) > 0:
    print(
        warning[
            [
                "engine_id",
                "cycle",
                "predicted_RUL",
                "maintenance_status"
            ]
        ].to_string(index=False)
    )
else:
    print("No warning engines found.")



# 12. Overall statistics


print("\n9. OVERALL STATISTICS")

print("-" * 60)

print(
    f"Total engines: {len(final_df)}"
)

print(
    f"Average predicted RUL: "
    f"{final_df['predicted_RUL'].mean():.2f}"
)

print(
    f"Minimum predicted RUL: "
    f"{final_df['predicted_RUL'].min():.2f}"
)

print(
    f"Maximum predicted RUL: "
    f"{final_df['predicted_RUL'].max():.2f}"
)



# 13. Final output

print("\n" + "=" * 60)
print("FINAL PREDICTION ANALYSIS COMPLETED")
print("=" * 60)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nNext step:")
print("Create dashboard-ready data for Power BI.")