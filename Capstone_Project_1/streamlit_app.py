# Capstone_Project_1/streamlit_app.py
import os
import json
from datetime import date, datetime
from io import StringIO

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# ---------- configuration ----------
BASE_DIR = os.path.dirname(__file__)
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "final_ridge_pipeline.joblib")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "feature_columns.json")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "model_metadata.json")

st.set_page_config(page_title="Manufacturing Output Predictor", layout="wide", page_icon="🏭")

# ---------- small CSS for nicer cards ----------
st.markdown(
    """
    <style>
    .big-result {
        background: linear-gradient(90deg,#114b29,#0d3b27);
        color: #fff;
        padding: 18px;
        border-radius: 12px;
        font-size: 20px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
    }
    .muted { color:#9aa0a6; font-size:12px }
    .card { background: #0f1113; padding:12px; border-radius:10px; color: #e6eef6; }
    .btn-red { color: #fff; background: #9b1e1e; padding:6px 10px; border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- helpers ----------
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error(f"Model not found at: {path}")
        st.stop()
    # Ensure custom transformer module is discoverable (we already included preprocessing_helpers)
    model = joblib.load(path)
    return model

def load_feature_columns(path=FEATURES_PATH):
    if os.path.exists(path):
        return json.load(open(path, "r"))
    return None

def load_metadata(path=METADATA_PATH):
    if os.path.exists(path):
        return json.load(open(path, "r"))
    return {}

def prepare_input_row(inputs: dict, feature_columns):
    """
    Produce a DataFrame with one row matching the model-feature order.
    Uses the same feature engineering as training (Total_Cycle_Time, ratios, date parts, ...)
    """
    d = dict(inputs)
    # Derived features
    d["Total_Cycle_Time"] = float(d.get("Cycle_Time", 0)) + float(d.get("Cooling_Time", 0))
    cycle = float(d.get("Cycle_Time", 0))
    cooling = float(d.get("Cooling_Time", 0))
    d["Cycle_Cooling_Ratio"] = cycle / (cooling + 1e-9)
    d["Temp_Pressure_Product"] = float(d.get("Injection_Temperature", 0)) * float(d.get("Injection_Pressure", 0))
    p = float(d.get("Injection_Pressure", 1))
    d["Temperature_Pressure_Ratio"] = float(d.get("Injection_Temperature", 0)) / (p + 1e-9)

    # date/time features
    sample_date = d.get("date", date.today())
    if isinstance(sample_date, str):
        sample_date = datetime.fromisoformat(sample_date).date()
    hour_val = int(d.get("hour", 0))
    dt = datetime.combine(sample_date, datetime.min.time()).replace(hour=hour_val)
    d["Hour"] = dt.hour
    d["Day"] = dt.day
    d["Month"] = dt.month
    d["Year"] = dt.year
    d["DayOfWeek_num"] = dt.weekday()

    # Keep categorical names used at training: Shift, Machine_Type, Material_Grade, Day_of_Week
    # Build DataFrame and ensure all expected columns exist
    df = pd.DataFrame([d])

    if feature_columns:
        for c in feature_columns:
            if c not in df.columns:
                df[c] = 0
        df = df[feature_columns]

    # convert numeric-like columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            pass

    return df

def model_predict(model, X):
    return float(model.predict(X)[0])

def compute_local_contributions(model, X_row, feature_columns):
    """
    For a linear/regression model that is Pipeline inside TransformedTargetRegressor,
    attempt to extract linear coefficients and compute coeff * value for each feature.
    Works if inner model is linear (e.g., Ridge).
    """
    try:
        # Unwrap TransformedTargetRegressor -> regressor -> pipeline -> model (Ridge)
        inner = model.regressor_
        lin = inner.named_steps["model"]
        preprocess = inner.named_steps["preprocess"]
        coef = lin.coef_
        # feature_columns order should align with coef length; otherwise we gracefully fallback
        if len(coef) == len(feature_columns):
            vals = X_row.values.flatten()
            contributions = dict(zip(feature_columns, (coef * vals).tolist()))
            # sort by absolute impact
            sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            return sorted_contrib[:8]  # top 8
    except Exception:
        pass
    return []

# ---------- app header ----------
st.title("🏭 Manufacturing Output Predictor")
st.markdown("Predict parts produced per hour from process settings — visual, fast and explainable.")

model = load_model()
feature_columns = load_feature_columns()
metadata = load_metadata()

# Top-level row: model metadata
col1, col2 = st.columns([3,1])
with col2:
    if metadata:
        st.markdown("**Model summary**")
        st.write(f"Type: {metadata.get('model_type', 'TransformedTargetRegressor')}")
        st.write(f"Saved: {metadata.get('timestamp', '-')}")
        if metadata.get("r2"):
            st.metric("R² (test)", f"{metadata['r2']:.3f}")
    else:
        st.markdown("**Model loaded**")
        st.write("Pipeline ready")

# ---------- main UI layout ----------
left, right = st.columns([1, 1.1])

with left:
    st.subheader("Input controls")
    # presets
    st.markdown("**Presets**")
    preset = st.radio("Scenario", ["Balanced", "High throughput (aggressive)", "Safe (conservative)"], index=0, horizontal=True)

    # default values (you can adjust)
    defaults = {
        "Balanced": {
            "Injection_Temperature": 220.0, "Injection_Pressure": 130.0, "Cycle_Time": 30.0, "Cooling_Time": 12.0,
            "Material_Viscosity": 300.0, "Ambient_Temperature": 25.0, "Machine_Age": 5, "Operator_Experience": 8,
            "Maintenance_Hours": 30, "Shift": "Evening", "Machine_Type": "Type_B", "Material_Grade": "Standard",
            "Day_of_Week": "Friday", "Efficiency_Score": 0.07, "Machine_Utilization": 0.55, "date": date.today(), "hour": 8
        },
        "High throughput (aggressive)": {
            "Injection_Temperature": 240.0, "Injection_Pressure": 150.0, "Cycle_Time": 25.0, "Cooling_Time": 8.0,
            "Material_Viscosity": 280.0, "Ambient_Temperature": 28.0, "Machine_Age": 3, "Operator_Experience": 6,
            "Maintenance_Hours": 20, "Shift": "Day", "Machine_Type": "Type_A", "Material_Grade": "Economy",
            "Day_of_Week": "Monday", "Efficiency_Score": 0.09, "Machine_Utilization": 0.85, "date": date.today(), "hour": 10
        },
        "Safe (conservative)": {
            "Injection_Temperature": 200.0, "Injection_Pressure": 120.0, "Cycle_Time": 35.0, "Cooling_Time": 15.0,
            "Material_Viscosity": 320.0, "Ambient_Temperature": 22.0, "Machine_Age": 7, "Operator_Experience": 10,
            "Maintenance_Hours": 40, "Shift": "Night", "Machine_Type": "Type_C", "Material_Grade": "Premium",
            "Day_of_Week": "Wednesday", "Efficiency_Score": 0.05, "Machine_Utilization": 0.4, "date": date.today(), "hour": 6
        },
    }

    vals = defaults[preset]

    # numeric inputs
    Injection_Temperature = st.number_input("Injection Temperature", value=vals["Injection_Temperature"], step=0.1)
    Injection_Pressure = st.number_input("Injection Pressure", value=vals["Injection_Pressure"], step=0.1)
    Cycle_Time = st.number_input("Cycle Time", value=vals["Cycle_Time"], step=0.1)
    Cooling_Time = st.number_input("Cooling Time", value=vals["Cooling_Time"], step=0.1)
    Material_Viscosity = st.number_input("Material Viscosity", value=vals["Material_Viscosity"], step=1.0)
    Ambient_Temperature = st.number_input("Ambient Temperature", value=vals["Ambient_Temperature"], step=0.1)
    Machine_Age = st.number_input("Machine Age (years)", value=vals["Machine_Age"], step=1.0)
    Operator_Experience = st.number_input("Operator Experience (years)", value=vals["Operator_Experience"], step=1.0)
    Maintenance_Hours = st.number_input("Maintenance Hours (per month)", value=vals["Maintenance_Hours"], step=1.0)
    Efficiency_Score = st.number_input("Efficiency Score", value=vals["Efficiency_Score"], step=0.01, format="%.2f")
    Machine_Utilization = st.number_input("Machine Utilization", value=vals["Machine_Utilization"], step=0.01, format="%.2f")
    sample_date = st.date_input("Sample date", value=vals["date"])
    hour_val = st.slider("Hour of Day (0-23)", 0, 23, vals["hour"])

    # categorical inputs
    Shift = st.selectbox("Shift", options=["Day", "Evening", "Night"], index=["Day","Evening","Night"].index(vals["Shift"]))
    Machine_Type = st.selectbox("Machine Type", options=["Type_A", "Type_B", "Type_C"], index=["Type_A","Type_B","Type_C"].index(vals.get("Machine_Type","Type_B")))
    Material_Grade = st.selectbox("Material Grade", options=["Economy", "Standard", "Premium"], index=["Economy","Standard","Premium"].index(vals.get("Material_Grade","Standard")))
    Day_of_Week = st.selectbox("Day of Week", options=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
                               index=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(vals.get("Day_of_Week","Friday")))

    # Predict button
    predict_clicked = st.button("🔮 Predict Output", key="predict_btn")

with right:
    st.subheader("Model insights")
    # show feature importance if available
    coeff_ax = None
    top_features_plot = st.empty()
    try:
        # get inner model coefficients (if available)
        inner = model.regressor_
        lin = inner.named_steps.get("model", None)
        preprocess = inner.named_steps.get("preprocess", None)
        if lin is not None and hasattr(lin, "coef_") and feature_columns:
            coefs = np.array(lin.coef_)
            # guard: lengths must match
            if coefs.size == len(feature_columns):
                coef_df = pd.DataFrame({"feature": feature_columns, "coef": coefs})
                coef_df["abs_coef"] = coef_df["coef"].abs()
                topn = coef_df.sort_values("abs_coef", ascending=False).head(12).iloc[::-1]
                fig, ax = plt.subplots(figsize=(6,4))
                ax.barh(topn["feature"], topn["coef"])
                ax.set_title("Top features (by coefficient magnitude)")
                ax.axvline(0, color="#444", linewidth=0.6)
                st.pyplot(fig)
            else:
                st.info("Model coefficients shape mismatch — skipping feature importance chart.")
        else:
            st.info("Feature importance not available for this model type.")
    except Exception:
        st.info("Couldn't compute feature importance.")

    st.write("---")
    st.markdown("<div class='card'><b>Quick tips</b><br>Try the presets to demonstrate model sensitivity. Use the download button after prediction to save your inputs + prediction.</div>", unsafe_allow_html=True)

# ---------- predictions, history and download ----------
if "history" not in st.session_state:
    st.session_state["history"] = []

if predict_clicked:
    raw_inputs = {
        "Injection_Temperature": Injection_Temperature,
        "Injection_Pressure": Injection_Pressure,
        "Cycle_Time": Cycle_Time,
        "Cooling_Time": Cooling_Time,
        "Material_Viscosity": Material_Viscosity,
        "Ambient_Temperature": Ambient_Temperature,
        "Machine_Age": Machine_Age,
        "Operator_Experience": Operator_Experience,
        "Maintenance_Hours": Maintenance_Hours,
        "Shift": Shift,
        "Machine_Type": Machine_Type,
        "Material_Grade": Material_Grade,
        "Day_of_Week": Day_of_Week,
        "Efficiency_Score": Efficiency_Score,
        "Machine_Utilization": Machine_Utilization,
        "date": sample_date.isoformat(),
        "hour": hour_val
    }

    with st.spinner("Preparing input and running model..."):
        X_pred = prepare_input_row(raw_inputs, feature_columns)
        try:
            pred_val = model_predict(model, X_pred)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write("Prepared input columns:", list(X_pred.columns))
            raise

    # baseline / thresholds from metadata if present
    baseline = metadata.get("mse", None)
    lower_warn = metadata.get("underperf_threshold", None)
    # Add to history
    rec = dict(raw_inputs)
    rec["predicted_parts_per_hour"] = pred_val
    rec["timestamp"] = datetime.now().isoformat()
    st.session_state.history.insert(0, rec)

    # Show result prominently
    colA, colB = st.columns([3,1])
    with colA:
        st.markdown(f"<div class='big-result'>✅  Predicted Output: <b style='font-size:24px'>{pred_val:.2f} units/hour</b></div>", unsafe_allow_html=True)
        if lower_warn:
            if pred_val < float(lower_warn):
                st.warning("⚠️ Predicted output below underperformance threshold — investigate inputs/maintenance.")
    with colB:
        st.metric("Pred (units/hr)", f"{pred_val:.2f}")

    # show top local contributors (linear approx)
    contribs = compute_local_contributions(model, X_pred, feature_columns or [])
    if contribs:
        st.subheader("Top contributors (approx)")
        for name, val in contribs:
            st.write(f"{name}: {val:.3f}")

    # provide download of input+prediction
    df_history = pd.DataFrame(st.session_state.history)
    csv = df_history.to_csv(index=False)
    st.download_button("Download history (CSV)", csv, "predictions_history.csv", "text/csv")

# History table
if st.session_state.history:
    st.subheader("Recent predictions")
    dfh = pd.DataFrame(st.session_state.history)
    st.dataframe(dfh.head(10), use_container_width=True)
