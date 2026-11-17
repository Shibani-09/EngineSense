import os
import pandas as pd


# ============================================================
# PREDICTIVE MAINTENANCE - DASHBOARD DATA
# ============================================================

print("\n" + "=" * 60)
print("PREDICTIVE MAINTENANCE - DASHBOARD DATA PREPARATION")
print("=" * 60)


# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

INPUT_FILE = "outputs/predictions/final_engine_predictions.csv"

OUTPUT_FILE = "powerbi/predictive_maintenance_dashboard.csv"


# ------------------------------------------------------------
# 2. Load final predictions
# ------------------------------------------------------------

print("\n1. Loading final engine predictions...")

df = pd.read_csv(INPUT_FILE)

print(f"   Rows: {len(df)}")
print(f"   Columns: {len(df.columns)}")


# ------------------------------------------------------------
# 3. Create priority level
# ------------------------------------------------------------

print("\n2. Creating maintenance priority...")

priority_map = {
    "Critical": 1,
    "Warning": 2,
    "Monitor": 3,
    "Healthy": 4
}

df["priority"] = df["maintenance_status"].map(priority_map)


# ------------------------------------------------------------
# 4. Create maintenance recommendation
# ------------------------------------------------------------

print("\n3. Creating maintenance recommendations...")


def recommendation(status):

    if status == "Critical":
        return "Immediate Maintenance Required"

    elif status == "Warning":
        return "Schedule Maintenance Soon"

    elif status == "Monitor":
        return "Continue Monitoring"

    else:
        return "No Immediate Action Required"


df["maintenance_recommendation"] = (
    df["maintenance_status"]
    .apply(recommendation)
)


# ------------------------------------------------------------
# 5. Create RUL category
# ------------------------------------------------------------

print("\n4. Creating RUL category...")


def rul_category(rul):

    if rul <= 30:
        return "0-30 Cycles"

    elif rul <= 60:
        return "31-60 Cycles"

    elif rul <= 120:
        return "61-120 Cycles"

    else:
        return "120+ Cycles"


df["RUL_category"] = df["predicted_RUL"].apply(
    rul_category
)


# ------------------------------------------------------------
# 6. Create risk percentage
# ------------------------------------------------------------

print("\n5. Creating risk percentage...")

df["risk_percentage"] = (
    (1 / df["predicted_RUL"]) * 100
).clip(upper=100)


df["risk_percentage"] = df["risk_percentage"].round(2)


# ------------------------------------------------------------
# 7. Sort by priority
# ------------------------------------------------------------

df = df.sort_values(
    ["priority", "predicted_RUL"],
    ascending=[True, True]
).reset_index(drop=True)


# ------------------------------------------------------------
# 8. Create Power BI directory
# ------------------------------------------------------------

os.makedirs("powerbi", exist_ok=True)


# ------------------------------------------------------------
# 9. Save dashboard dataset
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 10. Display summary
# ------------------------------------------------------------

print("\n6. DASHBOARD DATA SUMMARY")

print("-" * 60)

print(
    df[
        [
            "engine_id",
            "cycle",
            "predicted_RUL",
            "maintenance_status",
            "risk_score",
            "priority",
            "RUL_category",
            "maintenance_recommendation"
        ]
    ].head(20).to_string(index=False)
)


# ------------------------------------------------------------
# 11. Status summary
# ------------------------------------------------------------

print("\n7. MAINTENANCE STATUS")

print("-" * 60)

print(
    df["maintenance_status"]
    .value_counts()
)


# ------------------------------------------------------------
# 12. Overall statistics
# ------------------------------------------------------------

print("\n8. DASHBOARD KPIs")

print("-" * 60)

print(
    f"Total Engines       : {len(df)}"
)

print(
    f"Critical Engines    : "
    f"{(df['maintenance_status'] == 'Critical').sum()}"
)

print(
    f"Warning Engines     : "
    f"{(df['maintenance_status'] == 'Warning').sum()}"
)

print(
    f"Monitor Engines     : "
    f"{(df['maintenance_status'] == 'Monitor').sum()}"
)

print(
    f"Healthy Engines     : "
    f"{(df['maintenance_status'] == 'Healthy').sum()}"
)

print(
    f"Average Predicted RUL : "
    f"{df['predicted_RUL'].mean():.2f}"
)

print(
    f"Minimum Predicted RUL : "
    f"{df['predicted_RUL'].min():.2f}"
)


# ------------------------------------------------------------
# 13. Final message
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DASHBOARD DATA CREATED SUCCESSFULLY")
print("=" * 60)

print("\nPower BI file:")
print(OUTPUT_FILE)

print("\nNext step:")
print("Import this CSV into Power BI.")