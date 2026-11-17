import math
import re

import matplotlib.pyplot as plt
import streamlit as st

from streamlit_app.pages.engine_prediction import classify_engine_health
from streamlit_app.services.ml_service import (
    get_available_engines,
    load_feature_columns,
    load_test_dataset,
    predict_rul,
)


def show_realtime_monitoring() -> None:
    st.title("Simulated Engine Monitoring")
    st.warning(
        "SIMULATION ONLY — this page replays records from the processed "
        "NASA C-MAPSS FD001 test dataset. No live sensors or physical "
        "hardware are connected."
    )
    st.caption(
        "Each selected historical cycle is scored by the existing Random "
        "Forest using its already-engineered dataset features."
    )

    try:
        dataset = load_test_dataset()
        feature_columns = load_feature_columns()
        engine_ids = get_available_engines()
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Simulation data could not be loaded: {error}")
        return

    if not engine_ids:
        st.error("No engine records are available in the processed test data.")
        return

    selected_engine = st.selectbox(
        "Select engine to replay",
        engine_ids,
        format_func=lambda engine_id: f"Engine {engine_id}",
    )
    engine_rows = (
        dataset[dataset["engine_id"] == selected_engine]
        .sort_values("cycle")
        .reset_index(drop=True)
    )
    cycles = engine_rows["cycle"].astype(int).tolist()

    if not cycles:
        st.error(f"No processed observations found for engine {selected_engine}.")
        return

    selected_cycle = st.select_slider(
        "Replay dataset cycle",
        options=cycles,
        value=cycles[0],
    )
    observation = engine_rows.loc[
        engine_rows["cycle"] == selected_cycle
    ].iloc[-1]
    missing_features = [
        column for column in feature_columns if column not in observation.index
    ]
    if missing_features:
        st.error(
            "The selected dataset record is missing model features: "
            + ", ".join(missing_features)
        )
        return

    features = {
        column: float(observation[column])
        for column in feature_columns
    }
    try:
        predicted_rul = predict_rul(features)
        if not math.isfinite(predicted_rul):
            raise ValueError("The model returned a non-finite RUL value.")
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Simulation prediction failed: {error}")
        return

    health, recommendation = classify_engine_health(predicted_rul)
    metric_column, status_column = st.columns(2)
    metric_column.metric("Simulated RUL", f"{predicted_rul:.2f} cycles")
    status_column.metric("Health classification", health)
    st.info(recommendation)
    st.caption(
        f"Simulated observation · Engine {selected_engine} · "
        f"Dataset cycle {selected_cycle}"
    )

    sensor_columns = [
        column
        for column in engine_rows.columns
        if re.fullmatch(r"sensor_\d+", column)
    ]
    if not sensor_columns:
        st.warning("No raw sensor columns were found for this engine.")
        return

    selected_sensor = st.selectbox("Sensor trend", sensor_columns)
    replay_history = engine_rows[
        engine_rows["cycle"] <= selected_cycle
    ]
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(
        replay_history["cycle"],
        replay_history[selected_sensor],
        color="#1769E8",
    )
    axis.scatter(
        [selected_cycle],
        [float(observation[selected_sensor])],
        color="#E5A12B",
        zorder=3,
    )
    axis.set_xlabel("Cycle")
    axis.set_ylabel(selected_sensor)
    axis.set_title(
        f"Simulated {selected_sensor} history · Engine {selected_engine}"
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    with st.expander("Health classification thresholds"):
        st.markdown(
            "- **Critical:** predicted RUL ≤ 30 cycles\n"
            "- **Warning:** 30 < predicted RUL ≤ 60 cycles\n"
            "- **Monitor:** 60 < predicted RUL ≤ 120 cycles\n"
            "- **Healthy:** predicted RUL > 120 cycles"
        )
