import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from streamlit_app.services.ml_service import load_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_csv(path: Path, label: str, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} file was not found: {path.name}")

    try:
        dataframe = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} file could not be read: {error}") from error

    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"{label} file is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
    if dataframe.empty:
        raise ValueError(f"{label} file contains no records.")

    return dataframe


def _show_saved_chart(filename: str, title: str) -> None:
    chart_path = PROJECT_ROOT / "outputs" / "charts" / filename
    if chart_path.exists():
        st.image(str(chart_path), caption=title, width="stretch")
    else:
        st.warning(f"Saved chart is unavailable: {filename}")


def show_model_performance() -> None:
    st.title("Dataset Analysis & Model Performance")
    st.caption(
        "Exploration of the existing NASA C-MAPSS FD001 processed data and "
        "saved model evaluation results."
    )

    try:
        train_data = _read_csv(
            PROJECT_ROOT / "data" / "processed" / "train_features.csv",
            "Processed training dataset",
            {"engine_id", "cycle", "RUL"},
        )
        test_data = _read_csv(
            PROJECT_ROOT / "data" / "processed" / "test_features.csv",
            "Processed test dataset",
            {"engine_id", "cycle", "RUL"},
        )
    except (OSError, ValueError) as error:
        st.error(f"Dataset analysis is unavailable: {error}")
        return

    st.subheader("Processed dataset overview")
    train_col, test_col = st.columns(2)
    train_col.metric("Training records", f"{len(train_data):,}")
    train_col.metric("Training engines", f"{train_data['engine_id'].nunique():,}")
    test_col.metric("Test records", f"{len(test_data):,}")
    test_col.metric("Test engines", f"{test_data['engine_id'].nunique():,}")

    sensor_columns = [
        column
        for column in train_data.columns
        if re.fullmatch(r"sensor_\d+", column)
    ]
    try:
        feature_columns = load_feature_columns()
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Model feature metadata is unavailable: {error}")
        return

    missing_features = set(feature_columns).difference(test_data.columns)
    if missing_features:
        st.error(
            "Processed test data is missing model input columns: "
            + ", ".join(sorted(missing_features))
        )

    st.write(
        f"**Model input columns:** {len(feature_columns)} saved in the "
        f"trained model feature list. "
        f"**Raw sensor channels:** {len(sensor_columns)}."
    )
    st.caption(
        "RUL distributions below come from the supplied processed FD001 "
        "training and test labels; no feature engineering is run by this page."
    )

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.hist(
        train_data["RUL"].dropna(),
        bins=35,
        alpha=0.7,
        label="Training",
        color="#1769E8",
    )
    axis.hist(
        test_data["RUL"].dropna(),
        bins=35,
        alpha=0.55,
        label="Test",
        color="#8656D8",
    )
    axis.set_xlabel("Remaining useful life (cycles)")
    axis.set_ylabel("Records")
    axis.set_title("FD001 RUL distribution")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    st.subheader("Saved model evaluation")
    try:
        model_results = _read_csv(
            PROJECT_ROOT / "outputs" / "model_results.csv",
            "Model evaluation results",
            {"Model", "MAE", "RMSE", "R2"},
        )
        for _, row in model_results.iterrows():
            st.markdown(f"**{row['Model']}**")
            metric_columns = st.columns(3)
            for column, metric, value in zip(
                metric_columns,
                ("MAE", "RMSE", "R²"),
                (row["MAE"], row["RMSE"], row["R2"]),
            ):
                column.metric(metric, f"{float(value):.3f}")
    except (OSError, ValueError, TypeError) as error:
        st.error(f"Saved evaluation metrics are unavailable: {error}")

    left, right = st.columns(2)
    with left:
        _show_saved_chart(
            "actual_vs_predicted_rul.png",
            "Saved validation predictions: actual vs. predicted RUL",
        )
        _show_saved_chart(
            "model_comparison_mae.png",
            "Saved model comparison by MAE",
        )
        _show_saved_chart(
            "feature_importance.png",
            "Saved Random Forest feature importance",
        )
        _show_saved_chart(
            "rul_prediction_engine_2.png",
            "Saved actual vs. predicted RUL across validation engine cycles",
        )
    with right:
        _show_saved_chart(
            "prediction_error_distribution.png",
            "Saved validation prediction error distribution",
        )
        _show_saved_chart(
            "model_comparison_r2.png",
            "Saved model comparison by R²",
        )
