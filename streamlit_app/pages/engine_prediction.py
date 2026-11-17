import hashlib
import re
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pymongo.errors import PyMongoError

from streamlit_app.services.database import save_prediction_history
from streamlit_app.services.ml_service import (
    DATA_DIRECTORY,
    load_existing_dataset,
    load_uploaded_sensor_csv,
    list_existing_datasets,
    predict_existing_dataset,
    predict_uploaded_engines,
    validate_processed_dataset,
    validate_uploaded_sensor_data,
)


_HEALTH_RISK = {
    "Critical": "High",
    "Warning": "Elevated",
    "Monitor": "Moderate",
    "Healthy": "Low",
}


def classify_engine_health(predicted_rul: float) -> tuple[str, str]:
    """Classify model RUL using the application's documented thresholds."""

    if predicted_rul <= 30:
        return "Critical", "Immediate maintenance required"
    if predicted_rul <= 60:
        return "Warning", "Schedule maintenance soon"
    if predicted_rul <= 120:
        return "Monitor", "Continue monitoring the engine"
    return "Healthy", "No immediate action indicated"


def _decorate_results(results: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Add report metadata while keeping every original input row and column."""

    reserved_columns = {
        "health_status",
        "risk_category",
        "risk_score",
        "recommendation",
        "source_dataset",
        "prediction_timestamp",
    }
    collisions = reserved_columns.intersection(results.columns)
    if collisions:
        raise ValueError(
            "The input contains report output columns that would overwrite "
            "original data: " + ", ".join(sorted(collisions))
        )

    report = results.copy()
    classifications = report["predicted_rul"].apply(classify_engine_health)
    report["health_status"] = classifications.apply(lambda value: value[0])
    report["risk_category"] = report["health_status"].map(_HEALTH_RISK)
    report["risk_score"] = report["health_status"].map(
        {"Critical": 4, "Warning": 3, "Monitor": 2, "Healthy": 1}
    )
    report["recommendation"] = classifications.apply(lambda value: value[1])
    report["source_dataset"] = source_name
    report["prediction_timestamp"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    return report


def _latest_engine_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.sort_values(["engine_id", "cycle"])
        .groupby("engine_id", sort=True, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _save_history_once(
    results: pd.DataFrame,
    source_name: str,
    result_key: str,
) -> None:
    user = st.session_state.get("user")
    if not user or st.session_state.get("saved_prediction_history_key") == result_key:
        return

    latest = _latest_engine_results(results)
    try:
        save_prediction_history(
            user_id=user["id"],
            source_name=source_name,
            prediction_timestamp=str(results["prediction_timestamp"].iloc[0]),
            record_count=len(results),
            equipment_count=int(latest["engine_id"].nunique()),
            average_predicted_rul=float(latest["predicted_rul"].mean()),
            health_distribution={
                str(name): int(count)
                for name, count in latest["health_status"].value_counts().items()
            },
            report_csv=results.to_csv(index=False),
        )
    except (OSError, PyMongoError, RuntimeError, ValueError) as error:
        st.warning(
            "Predictions are ready, but their history could not be saved "
            f"({type(error).__name__}). Check the MongoDB configuration."
        )
    else:
        st.session_state["saved_prediction_history_key"] = result_key


def _show_prediction_report(
    results: pd.DataFrame,
    source_name: str,
    result_key: str,
) -> None:
    latest = _latest_engine_results(results)
    st.subheader("Actual model predictions")
    record_column, equipment_column, rul_column = st.columns(3)
    record_column.metric("Processed records", f"{len(results):,}")
    equipment_column.metric(
        "Equipment count", f"{latest['engine_id'].nunique():,}"
    )
    rul_column.metric(
        "Average latest RUL",
        f"{latest['predicted_rul'].mean():.2f} cycles",
    )
    st.caption(
        f"Source: {source_name} · "
        f"Prediction time (UTC): {results['prediction_timestamp'].iloc[0]}"
    )

    st.dataframe(
        latest[
            [
                "source_dataset",
                "engine_id",
                "cycle",
                "predicted_rul",
                "health_status",
                "risk_category",
                "prediction_timestamp",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    health_counts = latest["health_status"].value_counts().reindex(
        ["Critical", "Warning", "Monitor", "Healthy"], fill_value=0
    )
    chart_column, details_column = st.columns([1, 2])
    with chart_column:
        st.markdown("**Equipment health distribution**")
        st.bar_chart(health_counts, color="#1769E8")
    with details_column:
        with st.expander("All predicted input records"):
            st.dataframe(results, use_container_width=True, hide_index=True)

    st.download_button(
        "Download complete prediction report",
        data=results.to_csv(index=False).encode("utf-8-sig"),
        file_name="predictive_maintenance_predictions.csv",
        mime="text/csv",
        key=f"download_prediction_report_{result_key[:32]}",
        type="primary",
    )
    _save_history_once(results, source_name, result_key)


def _show_sensor_trend(dataframe: pd.DataFrame, key: str) -> None:
    sensor_columns = [
        column
        for column in dataframe.columns
        if re.fullmatch(r"sensor_\d+", str(column))
    ]
    if "engine_id" not in dataframe or "cycle" not in dataframe:
        return
    if not sensor_columns:
        st.info("No raw sensor columns are available for a trend chart.")
        return

    engines = sorted(dataframe["engine_id"].dropna().unique().tolist())
    if not engines:
        return
    engine_column, sensor_column = st.columns(2)
    selected_engine = engine_column.selectbox(
        "Equipment for sensor trend",
        engines,
        key=f"trend_engine_{key[:20]}",
    )
    selected_sensor = sensor_column.selectbox(
        "Sensor channel",
        sensor_columns,
        key=f"trend_sensor_{key[:20]}",
    )
    engine_rows = dataframe[dataframe["engine_id"] == selected_engine].sort_values(
        "cycle"
    )
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(
        engine_rows["cycle"],
        engine_rows[selected_sensor],
        color="#1769E8",
    )
    axis.set_xlabel("Cycle")
    axis.set_ylabel(selected_sensor)
    axis.set_title(f"{selected_sensor} readings · Engine {selected_engine}")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


def _run_existing_dataset_mode() -> None:
    st.subheader("Predict from an existing dataset")
    try:
        datasets = list_existing_datasets()
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Existing datasets could not be listed: {error}")
        return
    if not datasets:
        st.warning("No CSV datasets were found under the project data directory.")
        return

    default_dataset = next(
        (index for index, name in enumerate(datasets) if "test" in name.lower()),
        0,
    )
    selected_dataset = st.selectbox(
        "Available CSV dataset",
        datasets,
        index=default_dataset,
        key="existing_prediction_dataset",
    )
    dataset_path = (DATA_DIRECTORY / selected_dataset).resolve()
    try:
        signature = f"{selected_dataset}:{dataset_path.stat().st_mtime_ns}"
        dataframe = load_existing_dataset(selected_dataset)
        validated = validate_processed_dataset(dataframe)
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Dataset validation failed: {error}")
        return

    st.caption(f"Preview · {len(dataframe):,} records · {len(dataframe.columns)} columns")
    st.dataframe(dataframe.head(10), use_container_width=True, hide_index=True)
    st.success(
        f"Dataset valid: {validated['engine_id'].nunique():,} equipment units "
        f"and {len(validated):,} records."
    )

    result_key = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    if st.session_state.get("existing_prediction_key") != result_key:
        try:
            with st.spinner("Running the saved Random Forest model..."):
                predictions = predict_existing_dataset(validated)
                report = _decorate_results(predictions, selected_dataset)
        except (OSError, RuntimeError, ValueError) as error:
            st.session_state.pop("existing_prediction_report", None)
            st.error(f"Dataset prediction failed: {error}")
            return
        st.session_state["existing_prediction_report"] = report
        st.session_state["existing_prediction_key"] = result_key

    report = st.session_state.get("existing_prediction_report")
    if report is not None:
        _show_prediction_report(report, selected_dataset, result_key)
        st.subheader("Sensor history")
        _show_sensor_trend(validated, result_key)


def _run_uploaded_csv_mode() -> None:
    st.subheader("Upload a new sensor CSV")
    st.caption(
        "Upload a headered raw C-MAPSS CSV with engine IDs, cycles, three "
        "settings, and all 21 sensor columns. Each engine needs five "
        "observations with consecutive latest cycles. Predictions use only "
        "the readings provided; missing sensor values are never invented."
    )
    uploaded_file = st.file_uploader(
        "Choose a raw sensor CSV file",
        type=["csv"],
        key="engine_prediction_csv_upload",
    )
    if uploaded_file is None:
        return

    content = uploaded_file.getvalue()
    upload_signature = hashlib.sha256(content).hexdigest()
    result_key = f"upload:{upload_signature}"
    if st.session_state.get("uploaded_csv_signature") != upload_signature:
        st.session_state.pop("uploaded_csv_prediction_report", None)
        st.session_state["uploaded_csv_signature"] = upload_signature

    try:
        uploaded_data = load_uploaded_sensor_csv(uploaded_file.name, content)
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"CSV could not be read: {error}")
        return

    st.caption(
        f"Preview · {len(uploaded_data):,} records · "
        f"{len(uploaded_data.columns):,} columns"
    )
    st.dataframe(
        uploaded_data.head(10),
        use_container_width=True,
        hide_index=True,
    )
    try:
        validated_data = validate_uploaded_sensor_data(uploaded_data)
    except ValueError as error:
        st.error(f"CSV validation failed: {error}")
        return

    st.success(
        f"CSV valid: {validated_data['engine_id'].nunique():,} equipment units "
        f"and {len(validated_data):,} input records."
    )

    if st.session_state.get("uploaded_csv_result_key") != upload_signature:
        try:
            with st.spinner("Engineering features and running the saved model..."):
                predictions = predict_uploaded_engines(validated_data)
                report = _decorate_results(predictions, uploaded_file.name)
        except (OSError, RuntimeError, ValueError) as error:
            st.session_state.pop("uploaded_csv_prediction_report", None)
            st.error(f"Uploaded-data prediction failed: {error}")
            return
        st.session_state["uploaded_csv_prediction_report"] = report
        st.session_state["uploaded_csv_result_key"] = upload_signature

    report = st.session_state.get("uploaded_csv_prediction_report")
    if report is not None:
        _show_prediction_report(report, uploaded_file.name, result_key)
        st.subheader("Uploaded sensor history")
        _show_sensor_trend(validated_data, result_key)


def show_engine_prediction() -> None:
    st.title("Equipment RUL Prediction")
    st.caption(
        "Both modes run immediate batch inference with the saved Random Forest "
        "model. CSV inference is not continuous live sensor monitoring."
    )
    selected_mode = st.radio(
        "Prediction mode",
        ["Existing project dataset", "Upload a new CSV"],
        horizontal=True,
        key="prediction_mode",
    )
    if selected_mode == "Existing project dataset":
        _run_existing_dataset_mode()
    else:
        _run_uploaded_csv_mode()

    with st.expander("Health classification thresholds"):
        st.markdown(
            "- **Critical:** predicted RUL ≤ 30 cycles\n"
            "- **Warning:** 30 < predicted RUL ≤ 60 cycles\n"
            "- **Monitor:** 60 < predicted RUL ≤ 120 cycles\n"
            "- **Healthy:** predicted RUL > 120 cycles"
        )
        st.caption(
            "Risk and health labels are application-defined decision-support "
            "categories; they are not independent model outputs."
        )
