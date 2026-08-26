# -*- coding: utf-8 -*-

# ============================================================
# PHARMA - ANOMALY MONITORING DASHBOARD
# ============================================================
#
# Streamlit dashboard
#
# Layers integrated:
#   - Feature data
#   - Local IsolationForest anomaly detection
#   - FastAPI inference
#   - Streaming predictions monitoring
#   - Manual row diagnostic
#
# Home page is displayed by default.
# No row is selected automatically.
#
# ============================================================

from pathlib import Path
import json

import joblib
import pandas as pd
import requests
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PHARMA - Anomaly Monitoring",
    page_icon="P",
    layout="wide",
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURES_FILE = (
    ROOT
    / "data"
    / "processed"
    / "features.csv"
)

MODEL_FILE = (
    ROOT
    / "models"
    / "anomaly_model.joblib"
)

STREAM_FILE = (
    ROOT
    / "exports"
    / "stream_predictions.jsonl"
)

API_URL = "http://127.0.0.1:8000"


# ============================================================
# FEATURE CONTRACT
# ============================================================

ANOMALY_FEATURES = [
    "temperature",
    "ph",
    "dissolved_oxygen",
    "agitation_speed",
    "pressure",
    "flow_rate",
    "dissolved_oxygen_lag_1",
    "dissolved_oxygen_lag_2",
    "dissolved_oxygen_lag_3",
    "dissolved_oxygen_lag_5",
    "dissolved_oxygen_lag_10",
    "dissolved_oxygen_lag_15",
    "dissolved_oxygen_lag_30",
]


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #172B4D;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6B778C;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .normal-banner {
        background-color: #E3FCEF;
        border-left: 5px solid #00875A;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        color: #006644;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .anomaly-banner {
        background-color: #FFF0F0;
        border-left: 5px solid #DE350B;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        color: #BF2600;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD FEATURES DATA
# ============================================================

@st.cache_data
def load_data():

    if not FEATURES_FILE.exists():

        raise FileNotFoundError(
            f"Features file not found: {FEATURES_FILE}"
        )

    data = pd.read_csv(FEATURES_FILE)

    data["row_index"] = data.index

    if "timestamp" in data.columns:

        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce",
        )

    return data


# ============================================================
# LOAD ANOMALY MODEL
# ============================================================

@st.cache_resource
def load_anomaly_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Anomaly model not found: {MODEL_FILE}"
        )

    return joblib.load(MODEL_FILE)


# ============================================================
# CALCULATE LOCAL ANOMALIES
# ============================================================

@st.cache_data
def calculate_anomalies(data):

    model = load_anomaly_model()

    missing_features = [
        column
        for column in ANOMALY_FEATURES
        if column not in data.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing anomaly model features: "
            + ", ".join(missing_features)
        )

    X = data[ANOMALY_FEATURES].copy()

    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isnull().any().any():

        missing_count = int(
            X.isnull().sum().sum()
        )

        raise ValueError(
            "Anomaly feature matrix contains "
            f"{missing_count} missing/non-numeric values."
        )

    predictions = model.predict(X)

    scores = model.decision_function(X)

    result = data.copy()

    result["anomaly_prediction"] = predictions
    result["anomaly_score"] = scores
    result["is_anomaly"] = predictions == -1

    return result


# ============================================================
# FASTAPI HEALTH CHECK
# ============================================================

@st.cache_data(ttl=5)
def check_api():

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=2,
        )

        if response.status_code == 200:

            return response.json()

        return None

    except requests.RequestException:

        return None


# ============================================================
# FASTAPI PREDICTION
# ============================================================

def predict_with_api(row):

    payload = {
        "temperature": float(row["temperature"]),
        "ph": float(row["ph"]),
        "dissolved_oxygen": float(row["dissolved_oxygen"]),
        "agitation_speed": float(row["agitation_speed"]),
        "pressure": float(row["pressure"]),
        "flow_rate": float(row["flow_rate"]),
        "dissolved_oxygen_lag_1": float(
            row["dissolved_oxygen_lag_1"]
        ),
        "dissolved_oxygen_lag_2": float(
            row["dissolved_oxygen_lag_2"]
        ),
        "dissolved_oxygen_lag_3": float(
            row["dissolved_oxygen_lag_3"]
        ),
        "dissolved_oxygen_lag_5": float(
            row["dissolved_oxygen_lag_5"]
        ),
        "dissolved_oxygen_lag_10": float(
            row["dissolved_oxygen_lag_10"]
        ),
        "dissolved_oxygen_lag_15": float(
            row["dissolved_oxygen_lag_15"]
        ),
        "dissolved_oxygen_lag_30": float(
            row["dissolved_oxygen_lag_30"]
        ),
    }

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# LOAD STREAMING PREDICTIONS
# ============================================================

@st.cache_data
def load_stream_predictions():

    if not STREAM_FILE.exists():

        return pd.DataFrame()

    events = []

    with STREAM_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:

                event = json.loads(line)

                if isinstance(event, dict):

                    events.append(event)

            except json.JSONDecodeError:

                continue

    if not events:

        return pd.DataFrame()

    stream_df = pd.DataFrame(events)

    if "timestamp" in stream_df.columns:

        stream_df["timestamp"] = pd.to_datetime(
            stream_df["timestamp"],
            errors="coerce",
        )

    if "is_anomaly" in stream_df.columns:

        stream_df["is_anomaly"] = (
            stream_df["is_anomaly"]
            .astype(bool)
        )

    return stream_df


# ============================================================
# INITIALIZE DATA
# ============================================================

try:

    df = load_data()

    anomaly_df = calculate_anomalies(df)

except Exception as exc:

    st.error(
        "Unable to initialize anomaly dashboard:\n\n"
        f"{exc}"
    )

    st.stop()


# ============================================================
# LOAD STREAM
# ============================================================

stream_df = load_stream_predictions()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("PHARMA Controls")

equipment_options = sorted(
    anomaly_df["equipment_id"]
    .dropna()
    .astype(str)
    .unique()
)

selected_equipment = st.sidebar.selectbox(
    "Equipment",
    ["All"] + equipment_options,
    index=0,
)


# ============================================================
# FILTER MAIN DATA
# ============================================================

if selected_equipment == "All":

    filtered_df = anomaly_df.copy()

else:

    filtered_df = anomaly_df[
        anomaly_df["equipment_id"].astype(str)
        == selected_equipment
    ].copy()


# ============================================================
# FILTER STREAMING DATA
# ============================================================

if not stream_df.empty and "equipment_id" in stream_df.columns:

    if selected_equipment == "All":

        filtered_stream_df = stream_df.copy()

    else:

        filtered_stream_df = stream_df[
            stream_df["equipment_id"].astype(str)
            == selected_equipment
        ].copy()

else:

    filtered_stream_df = pd.DataFrame()


# ============================================================
# API STATUS
# ============================================================

api_status = check_api()

st.sidebar.subheader("System Status")

if api_status:

    st.sidebar.success("FastAPI: ONLINE")

    if api_status.get("anomaly_model"):

        st.sidebar.success(
            "Anomaly model: LOADED"
        )

    else:

        st.sidebar.warning(
            "Anomaly model: NOT LOADED"
        )

    if api_status.get("forecasting_model"):

        st.sidebar.success(
            "Forecasting model: LOADED"
        )

    else:

        st.sidebar.warning(
            "Forecasting model: NOT LOADED"
        )

else:

    st.sidebar.warning(
        "FastAPI: OFFLINE"
    )

    st.sidebar.caption(
        "Local anomaly analysis remains available."
    )


# ============================================================
# STREAMING STATUS
# ============================================================

st.sidebar.subheader("Streaming Status")

if STREAM_FILE.exists():

    st.sidebar.success(
        "Prediction stream: AVAILABLE"
    )

else:

    st.sidebar.warning(
        "Prediction stream: NOT AVAILABLE"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    "PHARMA - Real-Time AI"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Pharmaceutical process monitoring - "
    "anomaly detection, streaming inference and "
    "dissolved oxygen forecasting"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# ANOMALY MONITORING
# ============================================================

st.header("Anomaly Monitoring")

st.divider()


# ============================================================
# GLOBAL KPI
# ============================================================

total_rows = len(filtered_df)

total_anomalies = int(
    filtered_df["is_anomaly"].sum()
)

total_normal = (
    total_rows - total_anomalies
)

anomaly_rate = (
    total_anomalies / total_rows * 100
    if total_rows > 0
    else 0
)


kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:

    st.metric(
        "Rows analysed",
        f"{total_rows:,}",
    )

with kpi2:

    st.metric(
        "Anomalies",
        f"{total_anomalies:,}",
    )

with kpi3:

    st.metric(
        "Normal",
        f"{total_normal:,}",
    )

with kpi4:

    st.metric(
        "Anomaly rate",
        f"{anomaly_rate:.2f}%",
    )


# ============================================================
# STREAMING MONITORING
# ============================================================

st.header("Streaming Monitoring")

st.caption(
    "Live prediction events produced by the PHARMA streaming layer."
)


# ============================================================
# REFRESH STREAM
# ============================================================

refresh_col1, refresh_col2 = st.columns([1, 4])

with refresh_col1:

    if st.button(
        "Refresh streaming data",
        type="primary",
    ):

        load_stream_predictions.clear()

        st.rerun()


with refresh_col2:

    st.caption(
        f"Source: {STREAM_FILE}"
    )


# ============================================================
# STREAMING KPIs
# ============================================================

stream_total = len(filtered_stream_df)

if (
    not filtered_stream_df.empty
    and "is_anomaly" in filtered_stream_df.columns
):

    stream_anomalies = int(
        filtered_stream_df["is_anomaly"].sum()
    )

else:

    stream_anomalies = 0


stream_normal = (
    stream_total - stream_anomalies
)

stream_anomaly_rate = (
    stream_anomalies / stream_total * 100
    if stream_total > 0
    else 0
)


stream_kpi1, stream_kpi2, stream_kpi3, stream_kpi4 = (
    st.columns(4)
)

with stream_kpi1:

    st.metric(
        "Streaming events",
        f"{stream_total:,}",
    )

with stream_kpi2:

    st.metric(
        "Streaming anomalies",
        f"{stream_anomalies:,}",
    )

with stream_kpi3:

    st.metric(
        "Streaming normal",
        f"{stream_normal:,}",
    )

with stream_kpi4:

    st.metric(
        "Streaming anomaly rate",
        f"{stream_anomaly_rate:.2f}%",
    )


# ============================================================
# LAST STREAMING EVENT
# ============================================================

st.subheader("Latest streaming event")

if filtered_stream_df.empty:

    st.info(
        "No streaming prediction is available yet."
    )

else:

    latest_event = filtered_stream_df.iloc[-1]

    latest_equipment = latest_event.get(
        "equipment_id",
        "Unknown",
    )

    latest_timestamp = latest_event.get(
        "timestamp",
        "Unknown",
    )

    latest_score = latest_event.get(
        "anomaly_score",
        None,
    )

    latest_prediction = latest_event.get(
        "anomaly_prediction",
        None,
    )

    latest_is_anomaly = bool(
        latest_event.get(
            "is_anomaly",
            False,
        )
    )

    latest_oxygen = latest_event.get(
        "predicted_dissolved_oxygen",
        None,
    )


    if latest_is_anomaly:

        st.markdown(
            '<div class="anomaly-banner">'
            "ANOMALY DETECTED - Latest streaming event"
            "</div>",
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            '<div class="normal-banner">'
            "NORMAL - Latest streaming event"
            "</div>",
            unsafe_allow_html=True,
        )


    latest_col1, latest_col2, latest_col3, latest_col4, latest_col5 = (
        st.columns(5)
    )

    with latest_col1:

        st.metric(
            "Equipment",
            str(latest_equipment),
        )

    with latest_col2:

        st.metric(
            "Timestamp",
            str(latest_timestamp),
        )

    with latest_col3:

        if latest_score is not None:

            st.metric(
                "Anomaly score",
                f"{float(latest_score):.6f}",
            )

        else:

            st.metric(
                "Anomaly score",
                "N/A",
            )

    with latest_col4:

        if latest_oxygen is not None:

            st.metric(
                "Predicted dissolved O2",
                f"{float(latest_oxygen):.4f}",
            )

        else:

            st.metric(
                "Predicted dissolved O2",
                "N/A",
            )

    with latest_col5:

        st.metric(
            "Prediction",
            str(latest_prediction),
        )


# ============================================================
# STREAMING EVENT HISTORY
# ============================================================

st.subheader("Streaming event history")

if filtered_stream_df.empty:

    st.info(
        "No streaming events to display."
    )

else:

    stream_display_columns = [
        "equipment_id",
        "timestamp",
        "anomaly_prediction",
        "is_anomaly",
        "anomaly_score",
        "predicted_dissolved_oxygen",
    ]

    available_stream_columns = [
        column
        for column in stream_display_columns
        if column in filtered_stream_df.columns
    ]

    stream_history = filtered_stream_df[
        available_stream_columns
    ].copy()

    stream_history = stream_history.tail(100)

    st.dataframe(
        stream_history,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# ANOMALIES BY EQUIPMENT
# ============================================================

st.header("Anomalies by equipment")

if not filtered_df.empty:

    equipment_summary = (
        filtered_df
        .groupby("equipment_id")
        .agg(
            rows=("row_index", "count"),
            anomalies=("is_anomaly", "sum"),
        )
        .reset_index()
    )

    equipment_summary["anomaly_rate_%"] = (
        equipment_summary["anomalies"]
        / equipment_summary["rows"]
        * 100
    ).round(2)

    equipment_summary = equipment_summary.sort_values(
        "anomalies",
        ascending=False,
    )

    st.dataframe(
        equipment_summary,
        width="stretch",
        hide_index=True,
    )

else:

    st.info(
        "No data available for the selected equipment."
    )


# ============================================================
# DETECTED ANOMALIES
# ============================================================

st.header("Detected anomalies")

anomalies_only = filtered_df[
    filtered_df["is_anomaly"]
].copy()

anomalies_only = anomalies_only.sort_values(
    "anomaly_score",
    ascending=True,
)


display_columns = [
    "row_index",
    "equipment_id",
    "timestamp",
    "dissolved_oxygen",
    "anomaly_prediction",
    "anomaly_score",
]

available_display_columns = [
    column
    for column in display_columns
    if column in anomalies_only.columns
]

if anomalies_only.empty:

    st.success(
        "No anomalies detected for the selected equipment."
    )

else:

    st.dataframe(
        anomalies_only[
            available_display_columns
        ].head(100),
        width="stretch",
        hide_index=True,
    )

    if len(anomalies_only) > 100:

        st.caption(
            "Showing the 100 most anomalous rows "
            f"out of {len(anomalies_only):,} detected anomalies."
        )


# ============================================================
# PROCESS OVERVIEW
# ============================================================

st.header("Process Overview")

overview1, overview2, overview3, overview4 = (
    st.columns(4)
)

with overview1:

    st.metric(
        "Equipment",
        filtered_df["equipment_id"].nunique(),
    )

with overview2:

    mean_oxygen = filtered_df[
        "dissolved_oxygen"
    ].mean()

    st.metric(
        "Mean dissolved O2",
        f"{mean_oxygen:.3f}",
    )

with overview3:

    max_oxygen = filtered_df[
        "dissolved_oxygen"
    ].max()

    st.metric(
        "Max dissolved O2",
        f"{max_oxygen:.3f}",
    )

with overview4:

    st.metric(
        "Anomaly rate",
        f"{anomaly_rate:.2f}%",
    )


# ============================================================
# PROCESS VARIABLES
# ============================================================

st.header("Process Variables")

chart_columns = [
    "temperature",
    "ph",
    "dissolved_oxygen",
    "agitation_speed",
    "pressure",
    "flow_rate",
]

available_chart_columns = [
    column
    for column in chart_columns
    if column in filtered_df.columns
]

if available_chart_columns:

    chart_df = filtered_df[
        available_chart_columns
    ].copy()

    if len(chart_df) > 1500:

        chart_df = chart_df.tail(1500)

    st.line_chart(
        chart_df,
        width="stretch",
    )


# ============================================================
# DISSOLVED OXYGEN
# ============================================================

st.header("Dissolved Oxygen")

oxygen_columns = [
    "dissolved_oxygen",
    "dissolved_oxygen_lag_1",
    "dissolved_oxygen_lag_5",
    "dissolved_oxygen_lag_30",
]

available_oxygen_columns = [
    column
    for column in oxygen_columns
    if column in filtered_df.columns
]

if available_oxygen_columns:

    oxygen_df = filtered_df[
        available_oxygen_columns
    ].copy()

    if len(oxygen_df) > 1500:

        oxygen_df = oxygen_df.tail(1500)

    st.line_chart(
        oxygen_df,
        width="stretch",
    )


# ============================================================
# MANUAL ROW DIAGNOSTIC
# ============================================================

st.divider()

st.header("Row Diagnostic")

st.caption(
    "Select a row only if you want to inspect it in detail. "
    "No row is selected automatically."
)


# ============================================================
# ROW SELECTION
# ============================================================

if filtered_df.empty:

    st.info(
        "No rows available for diagnostic."
    )

else:

    row_options = (
        filtered_df["row_index"]
        .astype(int)
        .tolist()
    )

    selected_row_index = st.selectbox(
        "Select a row to inspect",
        options=row_options,
        index=None,
        placeholder="Choose a row...",
        key="diagnostic_row_selector",
    )

    open_diagnostic = st.button(
        "Open diagnostic",
        type="primary",
        disabled=selected_row_index is None,
    )


    # ========================================================
    # DIAGNOSTIC
    # ========================================================

    if (
        open_diagnostic
        and selected_row_index is not None
    ):

        selected_matches = filtered_df[
            filtered_df["row_index"]
            == selected_row_index
        ]

        if selected_matches.empty:

            st.error(
                "Selected row could not be found."
            )

        else:

            selected_row = (
                selected_matches.iloc[0]
            )

            row_number = int(
                selected_row["row_index"]
            )

            equipment = selected_row.get(
                "equipment_id",
                "Unknown",
            )

            prediction = int(
                selected_row["anomaly_prediction"]
            )

            score = float(
                selected_row["anomaly_score"]
            )

            is_anomaly = bool(
                selected_row["is_anomaly"]
            )


            # ====================================================
            # DIAGNOSTIC HEADER
            # ====================================================

            st.subheader(
                f"Diagnostic check - row {row_number}"
            )

            diagnostic1, diagnostic2, diagnostic3, diagnostic4 = (
                st.columns(4)
            )

            with diagnostic1:

                st.metric(
                    "Row",
                    row_number,
                )

            with diagnostic2:

                st.metric(
                    "Equipment",
                    equipment,
                )

            with diagnostic3:

                st.metric(
                    "Prediction",
                    prediction,
                )

            with diagnostic4:

                st.metric(
                    "Anomaly score",
                    f"{score:.6f}",
                )


            # ====================================================
            # STATUS
            # ====================================================

            if is_anomaly:

                st.error(
                    f"ROW {row_number} - ANOMALY DETECTED"
                )

            else:

                st.success(
                    f"ROW {row_number} - NORMAL"
                )


            # ====================================================
            # ROW INFORMATION
            # ====================================================

            st.subheader("Row information")

            info_columns = [
                "row_index",
                "equipment_id",
                "timestamp",
                "temperature",
                "ph",
                "dissolved_oxygen",
                "agitation_speed",
                "pressure",
                "flow_rate",
            ]

            available_info_columns = [
                column
                for column in info_columns
                if column in selected_row.index
            ]

            row_info = pd.DataFrame(
                {
                    "Feature": available_info_columns,
                    "Value": [
                        selected_row[column]
                        for column in available_info_columns
                    ],
                }
            )

            st.dataframe(
                row_info,
                width="stretch",
                hide_index=True,
            )


            # ====================================================
            # ANOMALY FEATURES
            # ====================================================

            st.subheader(
                "Anomaly model features"
            )

            feature_values = []

            for feature in ANOMALY_FEATURES:

                if feature in selected_row.index:

                    feature_values.append(
                        {
                            "Feature": feature,
                            "Value": selected_row[feature],
                        }
                    )

            feature_df = pd.DataFrame(
                feature_values
            )

            st.dataframe(
                feature_df,
                width="stretch",
                hide_index=True,
            )


            # ====================================================
            # LOCAL MODEL RESULT
            # ====================================================

            st.subheader(
                "Local anomaly model result"
            )

            model_result1, model_result2 = (
                st.columns(2)
            )

            with model_result1:

                st.metric(
                    "Prediction",
                    prediction,
                )

            with model_result2:

                st.metric(
                    "Anomaly score",
                    f"{score:.6f}",
                )


            # ====================================================
            # FASTAPI PREDICTION
            # ====================================================

            st.subheader(
                "AI inference"
            )

            if api_status:

                if st.button(
                    "Run FastAPI prediction",
                    key=f"api_predict_{row_number}",
                ):

                    try:

                        api_result = predict_with_api(
                            selected_row
                        )

                        st.success(
                            "FastAPI prediction completed."
                        )

                        result_col1, result_col2, result_col3 = (
                            st.columns(3)
                        )

                        with result_col1:

                            if (
                                "predicted_dissolved_oxygen"
                                in api_result
                            ):

                                st.metric(
                                    "Predicted dissolved O2",
                                    f"{api_result['predicted_dissolved_oxygen']:.4f}",
                                )

                        with result_col2:

                            if "anomaly_score" in api_result:

                                st.metric(
                                    "API anomaly score",
                                    f"{api_result['anomaly_score']:.4f}",
                                )

                        with result_col3:

                            if api_result.get(
                                "is_anomaly"
                            ):

                                st.error(
                                    "ANOMALY"
                                )

                            else:

                                st.success(
                                    "NORMAL"
                                )

                        with st.expander(
                            "Raw API response"
                        ):

                            st.json(
                                api_result
                            )

                    except Exception as exc:

                        st.error(
                            f"FastAPI prediction failed: {exc}"
                        )

            else:

                st.info(
                    "FastAPI is offline. "
                    "The local anomaly model is still available."
                )


# ============================================================
# RECENT FEATURE DATA
# ============================================================

st.divider()

st.header("Recent Feature Data")

recent_columns = [
    "row_index",
    "equipment_id",
    "timestamp",
    "temperature",
    "ph",
    "dissolved_oxygen",
    "agitation_speed",
    "pressure",
    "flow_rate",
    "anomaly_prediction",
    "anomaly_score",
]

available_recent_columns = [
    column
    for column in recent_columns
    if column in filtered_df.columns
]

st.dataframe(
    filtered_df[
        available_recent_columns
    ].tail(20),
    width="stretch",
    hide_index=True,
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "PHARMA - AI monitoring platform | "
    "Streamlit + FastAPI + IsolationForest + "
    "Scikit-learn + Streaming"
)

