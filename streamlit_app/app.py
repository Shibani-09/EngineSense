
import sys
from pathlib import Path

# Ensure the project root is available for imports on Streamlit Cloud.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import matplotlib as mpl

from pymongo.errors import PyMongoError

from streamlit_app.services.database import (
    get_database,
    get_prediction_history,
    is_auth_enabled,
    is_mongodb_configured,
)
from streamlit_app.services.authentication import register_user, authenticate_user
from streamlit_app.services.ml_service import (
    load_feature_columns,
    load_model,
    load_existing_dataset,
    list_existing_datasets,
    predict_existing_dataset,
)
from streamlit_app.pages.engine_prediction import show_engine_prediction
from streamlit_app.pages.model_performance import show_model_performance
from streamlit_app.pages.realtime_monitoring import show_realtime_monitoring


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

mpl.rcParams.update(
    {
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#DFE7F2",
        "axes.labelcolor": "#718096",
        "axes.titlecolor": "#1D2F4F",
        "xtick.color": "#718096",
        "ytick.color": "#718096",
        "text.color": "#1D2F4F",
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Arial"],
    }
)

st.markdown("""
<style>
    :root {
        color-scheme: light;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: #F4F7FC;
        color: #1D2F4F;
    }

    [data-testid="stHeader"] {
        background: rgba(244, 247, 252, 0.92);
        border-bottom: 1px solid rgba(223, 231, 242, 0.75);
    }

    .main .block-container {
        max-width: 1440px;
        padding: 2.4rem 3rem 3.5rem;
    }

    h1, h2, h3, h4 {
        color: #1D2F4F;
        letter-spacing: -0.025em;
    }

    h1 {
        font-weight: 750;
        line-height: 1.2;
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: #718096;
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #092653 0%, #0B3268 100%);
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #F7FAFF;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #B5C9E8;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        min-height: 2.8rem;
        margin: 0.2rem 0;
        padding: 0.55rem 0.8rem;
        border-radius: 0.7rem;
        transition: background-color 160ms ease, transform 160ms ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.09);
        transform: translateX(2px);
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(110deg, #1769E8 0%, #3486F2 100%);
        box-shadow: 0 7px 18px rgba(2, 20, 52, 0.24);
    }

    [data-testid="stSidebar"] [data-testid="stButton"] button {
        color: #FFFFFF;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.22);
    }

    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        background: rgba(255, 255, 255, 0.18);
        border-color: rgba(255, 255, 255, 0.42);
    }

    div[data-testid="stMetric"] {
        min-height: 112px;
        padding: 1.15rem 1.3rem;
        
        background: #FFFFFF;
        border: 1px solid #DFE7F2;
        border-left: 4px solid #1769E8;
        border-radius: 1rem;
        box-shadow: 0 8px 24px rgba(20, 47, 86, 0.07);
        transition: transform 180ms ease, box-shadow 180ms ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 30px rgba(20, 47, 86, 0.12);
    }

    [data-testid="stMetricLabel"] {
        color: #718096;
        font-size: 0.88rem;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #1D2F4F;
        font-weight: 750;
    }

    [data-testid="stButton"] button,
    [data-testid="stFormSubmitButton"] button {
        min-height: 2.7rem;
        border-radius: 0.72rem;
        font-weight: 650;
        transition: transform 160ms ease, box-shadow 160ms ease,
            background-color 160ms ease, border-color 160ms ease;
    }

    [data-testid="stButton"] button[kind="primary"],
    [data-testid="stFormSubmitButton"] button[kind="primary"] {
        color: #FFFFFF;
        background: linear-gradient(115deg, #1769E8 0%, #244A91 100%);
        border: 1px solid #1769E8;
        box-shadow: 0 7px 16px rgba(23, 105, 232, 0.23);
    }

    [data-testid="stButton"] button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
        background: linear-gradient(115deg, #3486F2 0%, #1769E8 100%);
        border-color: #1769E8;
        box-shadow: 0 10px 22px rgba(23, 105, 232, 0.3);
        transform: translateY(-1px);
    }

    [data-testid="stButton"] button:not([kind="primary"]),
    [data-testid="stFormSubmitButton"] button:not([kind="primary"]) {
        color: #244A91;
        background: #FFFFFF;
        border: 1px solid #DFE7F2;
    }

    [data-testid="stButton"] button:not([kind="primary"]):hover,
    [data-testid="stFormSubmitButton"] button:not([kind="primary"]):hover {
        color: #1769E8;
        border-color: #1769E8;
        box-shadow: 0 6px 16px rgba(23, 105, 232, 0.12);
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        color: #1D2F4F;
        background: #FFFFFF;
        border-color: #DFE7F2;
        border-radius: 0.7rem;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: #1769E8;
        box-shadow: 0 0 0 3px rgba(23, 105, 232, 0.12);
    }

    [data-testid="stForm"],
    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #DFE7F2;
        border-radius: 1rem;
        box-shadow: 0 8px 24px rgba(20, 47, 86, 0.06);
    }

    [data-testid="stForm"] {
        padding: 1.25rem 1.4rem;
    }

    [data-testid="stExpander"] summary {
        color: #1D2F4F;
        font-weight: 650;
    }

    [data-baseweb="tab-list"] {
        gap: 0.45rem;
        border-bottom: 1px solid #DFE7F2;
    }

    [data-testid="stTab"] {
        color: #718096;
        background: transparent;
        border-radius: 0.65rem 0.65rem 0 0;
        font-weight: 650;
        transition: color 160ms ease, background-color 160ms ease;
    }

    [data-testid="stTab"] p {
        color: inherit;
    }

    [data-testid="stTab"][aria-selected="true"] {
        color: #1769E8;
        background: #E5EEFF;
        border-color: #1769E8;
    }

    [data-testid="stTab"][aria-selected="true"] .react-aria-SelectionIndicator {
        background: #1769E8;
    }

    [data-testid="stAlert"] {
        border-radius: 0.85rem;
        border: 1px solid #DFE7F2;
        box-shadow: 0 5px 16px rgba(20, 47, 86, 0.04);
    }

    [data-testid="stPyplot"],
    [data-testid="stPlotlyChart"],
    [data-testid="stImage"] {
        overflow: hidden;
        padding: 0.5rem;
        background: #FFFFFF;
        border: 1px solid #DFE7F2;
        border-radius: 1rem;
        box-shadow: 0 8px 24px rgba(20, 47, 86, 0.07);
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        overflow: hidden;
        border: 1px solid #DFE7F2;
        border-radius: 0.85rem;
        box-shadow: 0 6px 18px rgba(20, 47, 86, 0.05);
    }

    hr {
        border-color: #DFE7F2;
    }

    @media (max-width: 900px) {
        .main .block-container {
            padding: 1.8rem 1.4rem 2.5rem;
        }
    }

    @media (max-width: 600px) {
        .main .block-container {
            padding: 1.35rem 0.9rem 2rem;
        }

        div[data-testid="stMetric"] {
            min-height: auto;
            padding: 0.9rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE AND CACHED DASHBOARD DATA
# --------------------------------------------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None


@st.cache_data(show_spinner=False)
def _dashboard_prediction(dataset_name: str, modified_time: int) -> pd.DataFrame:
    del modified_time
    return predict_existing_dataset(load_existing_dataset(dataset_name))


def _health_status(predicted_rul: float) -> str:
    from streamlit_app.pages.engine_prediction import classify_engine_health

    return classify_engine_health(float(predicted_rul))[0]


# --------------------------------------------------
# AUTHENTICATION PAGE
# --------------------------------------------------

def show_authentication() -> None:
    st.title("Predictive Maintenance")
    st.caption("Industrial equipment health monitoring")
    st.info("Sign in to access protected predictions and personal history.")

    login_tab, register_tab = st.tabs(["Login", "Create account"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email address", autocomplete="email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Login", type="primary", use_container_width=True
            )
        if submitted:
            try:
                success, message, user = authenticate_user(email, password)
            except (OSError, PyMongoError, RuntimeError, ValueError) as error:
                st.error(
                    "Login could not be completed "
                    f"({type(error).__name__}). Check MongoDB settings."
                )
            else:
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    st.rerun()
                st.error(message)

    with register_tab:
        with st.form("registration_form"):
            full_name = st.text_input("Full name", max_chars=120)
            email = st.text_input("Email address", autocomplete="email")
            password = st.text_input(
                "Password", type="password", help="At least 8 characters."
            )
            confirm_password = st.text_input(
                "Confirm password", type="password"
            )
            submitted = st.form_submit_button(
                "Create account", type="primary", use_container_width=True
            )
        if submitted:
            if password != confirm_password:
                st.error("Passwords do not match.")
            else:
                try:
                    success, message = register_user(full_name, email, password)
                except (OSError, PyMongoError, RuntimeError, ValueError) as error:
                    st.error(
                        "Registration could not be completed "
                        f"({type(error).__name__}). Check MongoDB settings."
                    )
                else:
                    if success:
                        st.success(message)
                        st.info("Your account is ready. You can now log in.")
                    else:
                        st.error(message)


def _show_sidebar() -> str:
    st.sidebar.title("Predictive Maintenance")
    st.sidebar.caption("Industrial equipment monitoring")
    user = st.session_state.get("user")
    if user:
        st.sidebar.write(f"Welcome, {user['full_name']}")
        st.sidebar.caption(user["email"])
        if st.sidebar.button("Sign out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()

    pages = [
        "Home Dashboard",
        "Engine Prediction",
        "Model Performance",
        "Realtime Monitoring",
    ]
    if user:
        pages.append("Prediction History")
    return st.sidebar.radio("Navigation", pages)


# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

def show_prediction_history() -> None:
    st.title("Prediction History")
    user = st.session_state.get("user")
    if not user:
        st.error("Sign in to view prediction history.")
        return

    try:
        history = get_prediction_history(user["id"], limit=50)
    except (OSError, PyMongoError, RuntimeError, ValueError) as error:
        st.error(
            "Prediction history is unavailable "
            f"({type(error).__name__}). Check MongoDB settings."
        )
        return

    if not history:
        st.info("No predictions have been saved for this account yet.")
        return

    for item in history:
        heading = (
            f"{item['source_name']} · {item['prediction_timestamp']} "
            f"(UTC)"
        )
        with st.expander(heading):
            total_column, equipment_column, rul_column = st.columns(3)
            total_column.metric("Processed records", item["record_count"])
            equipment_column.metric("Equipment", item["equipment_count"])
            rul_column.metric(
                "Average latest RUL",
                f"{item['average_predicted_rul']:.2f} cycles",
            )
            st.write("**Health distribution**", item["health_distribution"])
            if item.get("report_csv") is not None:
                st.download_button(
                    "Download saved report",
                    data=item["report_csv"].encode("utf-8-sig"),
                    file_name=f"prediction-history-{item['id'][:8]}.csv",
                    mime="text/csv",
                    key=f"history_download_{item['id']}",
                )
            else:
                st.caption(
                    "The report exceeded the 256 KiB history-storage limit. "
                    "Prediction metadata is retained; the original records "
                    "were not stored."
                )


# --------------------------------------------------
# AUTHENTICATED AND PUBLIC APPLICATION PAGES
# --------------------------------------------------

def show_dashboard() -> None:
    selected_page = _show_sidebar()
    if selected_page == "Engine Prediction":
        show_engine_prediction()
        return
    if selected_page == "Model Performance":
        show_model_performance()
        return
    if selected_page == "Realtime Monitoring":
        show_realtime_monitoring()
        return
    if selected_page == "Prediction History":
        show_prediction_history()
        return

    st.title("Predictive Maintenance Dashboard")
    st.caption(
        "Actual Random Forest inference against the saved NASA C-MAPSS "
        "FD001 test dataset."
    )

    database_status = "Not configured"
    database_name = None
    if is_mongodb_configured():
        try:
            database = get_database()
        except (OSError, PyMongoError, RuntimeError, ValueError) as error:
            database_status = f"Unavailable ({type(error).__name__})"
        else:
            database_status = "Connected"
            database_name = database.name
    elif is_auth_enabled():
        st.error("Authentication is enabled, but MongoDB is not configured.")

    try:
        datasets = list_existing_datasets()
        test_dataset = next(
            (name for name in datasets if "test" in name.lower()),
            None,
        )
        if test_dataset is None:
            raise FileNotFoundError("No processed test CSV was found under data/.")
        dataset_path = PROJECT_ROOT / "data" / test_dataset
        dashboard_results = _dashboard_prediction(
            test_dataset, dataset_path.stat().st_mtime_ns
        )
        latest = (
            dashboard_results.sort_values(["engine_id", "cycle"])
            .groupby("engine_id", sort=True, as_index=False)
            .tail(1)
            .copy()
        )
        latest["health_status"] = latest["predicted_rul"].apply(_health_status)
        latest["risk_category"] = latest["health_status"].map(
            {
                "Critical": "High",
                "Warning": "Elevated",
                "Monitor": "Moderate",
                "Healthy": "Low",
            }
        )
        feature_count = len(load_feature_columns())
        load_model()
    except (OSError, RuntimeError, ValueError) as error:
        dashboard_results = None
        latest = None
        feature_count = 0
        st.error(f"Dashboard model data is unavailable: {error}")

    record_count = len(dashboard_results) if dashboard_results is not None else None
    equipment_count = latest["engine_id"].nunique() if latest is not None else None
    average_rul = (
        float(latest["predicted_rul"].mean()) if latest is not None else None
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Processed records",
        f"{record_count:,}" if record_count is not None else "Unavailable",
    )
    metric_columns[1].metric(
        "Equipment",
        f"{equipment_count:,}" if equipment_count is not None else "Unavailable",
    )
    metric_columns[2].metric(
        "Average latest RUL",
        f"{average_rul:.2f} cycles" if average_rul is not None else "Unavailable",
    )
    metric_columns[3].metric("Model features", feature_count or "Unavailable")
    st.caption(
        f"MongoDB: {database_status}"
        + (f" · Database: {database_name}" if database_name else "")
    )

    if latest is not None:
        health_counts = latest["health_status"].value_counts().reindex(
            ["Critical", "Warning", "Monitor", "Healthy"], fill_value=0
        )
        risk_counts = latest["risk_category"].value_counts().reindex(
            ["High", "Elevated", "Moderate", "Low"], fill_value=0
        )
        health_column, risk_column = st.columns(2)
        with health_column:
            st.subheader("Equipment health")
            st.bar_chart(health_counts, color="#1769E8")
        with risk_column:
            st.subheader("Equipment risk")
            st.bar_chart(risk_counts, color="#E5A12B")

    metrics_path = PROJECT_ROOT / "outputs" / "final_model_metrics.csv"
    if metrics_path.is_file():
        try:
            metrics_df = pd.read_csv(metrics_path)
            saved_metrics = {
                str(row["Metric"]): float(row["Value"])
                for _, row in metrics_df.iterrows()
                if {"Metric", "Value"}.issubset(metrics_df.columns)
            }
        except (OSError, pd.errors.ParserError, TypeError, ValueError) as error:
            st.warning(f"Saved model metrics could not be read: {error}")
        else:
            if saved_metrics:
                st.subheader("Saved validation metrics")
                columns = st.columns(len(saved_metrics))
                for column, (name, value) in zip(columns, saved_metrics.items()):
                    column.metric(name, f"{value:.3f}")
                st.caption(
                    "These are saved validation results for the existing model; "
                    "the model is not retrained on app startup."
                )

    user = st.session_state.get("user")
    if user and database_status == "Connected":
        try:
            history = get_prediction_history(
                user["id"], limit=10, include_reports=False
            )
        except (OSError, PyMongoError, RuntimeError, ValueError) as error:
            st.warning(
                "Personal prediction trends are unavailable "
                f"({type(error).__name__})."
            )
        else:
            if len(history) > 1:
                history_frame = pd.DataFrame(history)
                history_frame["prediction_timestamp"] = pd.to_datetime(
                    history_frame["prediction_timestamp"], utc=True, errors="coerce"
                )
                history_frame = history_frame.dropna(
                    subset=["prediction_timestamp"]
                ).sort_values("prediction_timestamp")
                if not history_frame.empty:
                    st.subheader("Your prediction trend")
                    st.line_chart(
                        history_frame.set_index("prediction_timestamp")[
                            "average_predicted_rul"
                        ]
                    )

    st.caption(
        "The monitoring page replays historical dataset rows. No live sensors "
        "or continuous streaming source are connected."
    )


# --------------------------------------------------
# APPLICATION ENTRY POINT
# --------------------------------------------------

auth_enabled = is_auth_enabled()
if auth_enabled and not is_mongodb_configured():
    st.error(
        "AUTH_ENABLED is true but MongoDB is not configured. Set MONGODB_URI "
        "and MONGODB_DATABASE before enabling login."
    )
    st.stop()

if auth_enabled and not (
    st.session_state.authenticated and st.session_state.user
):
    st.session_state.authenticated = False
    st.session_state.user = None
    page = st.navigation(
        [st.Page(show_authentication, title="Login", default=True)],
        position="hidden",
    )
else:
    if not auth_enabled:
        st.session_state.authenticated = False
        st.session_state.user = None
    page = st.navigation(
        [st.Page(show_dashboard, title="Predictive Maintenance", default=True)],
        position="hidden",
    )

page.run()
