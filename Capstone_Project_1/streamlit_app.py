# Capstone_Project_1/streamlit_app.py
# Final Streamlit app for Manufacturing Output Prediction
# - robust import for QuantileClipper so joblib unpickle works
# - safe UI defaults
# - prepare_input_row returns raw categorical columns the pipeline expects

import os
import sys
import logging
import json
from datetime import date, datetime, time

# simple diagnostic writer (Streamlit Cloud logs will include this file content)
def write_diag(msg):
    try:
        logging.error(msg)
        with open("/tmp/streamlit_startup_diagnostic.txt", "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass

write_diag("==== STARTUP DIAGNOSTIC ====")
write_diag("CWD: " + os.getcwd())
try:
    write_diag("LISTING CWD: " + str(os.listdir(".")))
except Exception as e:
    write_diag("LISTING CWD FAILED: " + str(e))
write_diag("sys.path (first 15 entries):")
for p in sys.path[:15]:
    write_diag("  " + str(p))

# Try to import QuantileClipper from likely locations so joblib.load can find it when unpickling
QuantileClipper = None
_import_errors = {}

try:
    from Capstone_Project_1.preprocessing_helpers import QuantileClipper as QC_pkg
    QuantileClipper = QC_pkg
    write_diag("Imported QuantileClipper via Capstone_Project_1.preprocessing_helpers")
except Exception as e:
    _import_errors["pkg"] = repr(e)
    write_diag("Failed import Capstone_Project_1.preprocessing_helpers: " + repr(e))

if QuantileClipper is None:
    try:
        from preprocessing_helpers import QuantileClipper as QC_local
        QuantileClipper = QC_local
        write_diag("Imported QuantileClipper via preprocessing_helpers (local)")
    except Exception as e:
        _import_errors["local"] = repr(e)
        write_diag("Failed import preprocessing_helpers: " + repr(e))

# try a few alternate package name casings if still missing
if QuantileClipper is None:
    for alt in ["capstone_project_1", "Capstone_project_1", "capstone_Project_1", "Capstone_Project_1"]:
        try:
            mod = __import__(f"{alt}.preprocessing_helpers", fromlist=["preprocessing_helpers"])
            QC_alt = getattr(mod, "QuantileClipper", None)
            if QC_alt:
                QuantileClipper = QC_alt
                write_diag(f"Imported QuantileClipper via alt package {alt}")
                break
        except Exception as e:
            _import_errors[f"alt_{alt}"] = repr(e)

if QuantileClipper is None:
    write_diag("All QuantileClipper import attempts failed. Summary: " + json.dumps(_import_errors))
    raise ModuleNotFoundError(
        "QuantileClipper import failed. Ensure Capstone_Project_1/preprocessing_helpers.py exists and contains QuantileClipper."
    )

# expose to globals so joblib unpickle can find it
globals()["QuantileClipper"] = QuantileClipper

# normal imports
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# configuration
BASE_DIR = os.path.dirname(__file__)
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "final_ridge_pipeline.joblib")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "feature_columns.json")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "model_metadata.json")

st.set_page_config(page_title="Manufacturing Output Predictor", layout="wide", page_icon="🏭")

st.markdown(
    """
    <style>
    .big-result { background: linear-gradient(90deg,#114b29,#0d3b27); color: #fff; padding: 18px; border-radius: 12px; font-size: 20px; box-shadow: 0 6px 18px rgba(0,0,0,0.3); }
    .card { background: #0f1113; padding:12px; border-radius:10px; color: #e6eef6; }
    </style>
    """, unsafe_allow_html=True)

# helpers
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error(f"Model not found at: {path}")
        st.stop()
    model = joblib.load(path)  # QuantileClipper available in globals()
    return model

def load_feature_columns(path=FEATURES_PATH):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def load_metadata(path=METADATA_PATH):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def prepare_input_row(inputs, feature_columns=None):
    """
    Build a single-row DataFrame containing the RAW columns the saved pipeline expects.
    IMPORTANT: Do NOT pre-one-hot encode categorical columns here. ColumnTransformer in the pipeline
    expects raw categorical columns like 'Shift', 'Machine_Type', 'Material_Grade', 'Day_of_Week'.
    """
    d = dict(inputs)

    # engineered numeric features (match training notebook)
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
        try:
            sample_date = datetime.fromisoformat(sample_date).date()
        except Exception:
            sample_date = date.today()
    hour_val = int(d.get("hour", 0))
    try:
        dt = datetime.combine(sample_date, time(hour=hour_val))
    except Exception:
        dt = datetime.now()
    d["Hour"] = dt.hour
    d["Day"] = dt.day
    d["Month"] = dt.month
    d["Year"] = dt.year
    d["DayOfWeek_num"] = dt.weekday()

    # Ensure raw categorical columns exist (strings) — ColumnTransformer will handle encoding
    d["Shift"] = str(d.get("Shift", "Day"))
    d["Machine_Type"] = str(d.get("Machine_Type", "Type_A"))
    d["Material_Grade"] = str(d.get("Material_Grade", "Standard"))
    d["Day_of_Week"] = str(d.get("Day_of_Week", "Monday"))

    # Build dataframe using raw inputs + engineered numerics
    df = pd.DataFrame([d])

    # Ensure numeric columns exist and convert where possible
    numeric_like = [
        "Injection_Temperature", "Injection_Pressure", "Cycle_Time", "Cooling_Time",
        "Material_Viscosity", "Ambient_Temperature", "Machine_Age", "Operator_Experience",
        "Maintenance_Hours", "Temperature_Pressure_Ratio", "Total_Cycle_Time",
        "Efficiency_Score", "Machine_Utilization", "Hour", "Day", "Month", "Year",
        "DayOfWeek_num", "Temp_Pressure_Product", "Cycle_Cooling_Ratio"
    ]
    for col in numeric_like:
        if col not in df.columns:
            df[col] = 0
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass

    # Ensure categorical columns are present and string type
    for cat_col in ["Shift", "Machine_Type", "Material_Grade", "Day_of_Week"]:
        if cat_col not in df.columns:
            df[cat_col] = ""
        df[cat_col] = df[cat_col].astype(str)

    # Do NOT reorder to feature_columns (feature_columns contains post-transform names)
    return df

def model_predict(model, X):
    return float(model.predict(X)[0])

def compute_local_contributions(model, X_row, feature_columns):
    try:
        inner = model.regressor_
        lin = inner.named_steps.get("model", None)
        if lin is None or not hasattr(lin, "coef_") or not feature_columns:
            return []
        coef = np.array(lin.coef_)
        if len(coef) != len(feature_columns):
            return []
        vals = X_row.values.flatten()
        contributions = dict(zip(feature_columns, (coef * vals).tolist()))
        sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_contrib[:8]
    except Exception:
        return []

# App header
st.title("🏭 Manufacturing Output Predictor")
st.markdown("Predict parts produced per hour from process settings — visual, fast and explainable.")

# Load model and artifacts
feature_columns = load_feature_columns()
metadata = load_metadata()
try:
    model = load_model()
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.write("Check the startup diagnostics in logs (/tmp/streamlit_startup_diagnostic.txt).")
    st.stop()

# Layout: left inputs, right insights
left, right = st.columns([1, 1.05])

with left:
    st.subheader("Input controls")

    preset = st.radio("Scenario", ["Balanced", "High throughput (aggressive)", "Safe (conservative)"],
                      index=0, horizontal=True)

    # safe defaults
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

    vals = defaults.get(preset, defaults["Balanced"]) or defaults["Balanced"]

    def safe_num(key, default):
        try:
            return float(vals.get(key, default))
        except Exception:
            return float(default)

    def safe_cat(key, options, default):
        v = vals.get(key, default)
        return v if v in options else options[0]

    Injection_Temperature = st.number_input("Injection Temperature", value=safe_num("Injection_Temperature", 220.0), step=0.1)
    Injection_Pressure = st.number_input("Injection Pressure", value=safe_num("Injection_Pressure", 130.0), step=0.1)
    Cycle_Time = st.number_input("Cycle Time", value=safe_num("Cycle_Time", 30.0), step=0.1)
    Cooling_Time = st.number_input("Cooling Time", value=safe_num("Cooling_Time", 12.0), step=0.1)
    Material_Viscosity = st.number_input("Material Viscosity", value=safe_num("Material_Viscosity", 300.0), step=1.0)
    Ambient_Temperature = st.number_input("Ambient Temperature", value=safe_num("Ambient_Temperature", 25.0), step=0.1)
    Machine_Age = st.number_input("Machine Age (years)", value=safe_num("Machine_Age", 5.0), step=1.0)
    Operator_Experience = st.number_input("Operator Experience (years)", value=safe_num("Operator_Experience", 5.0), step=1.0)
    Maintenance_Hours = st.number_input("Maintenance Hours (per month)", value=safe_num("Maintenance_Hours", 20.0), step=1.0)
    Efficiency_Score = st.number_input("Efficiency Score", value=safe_num("Efficiency_Score", 0.05), step=0.01, format="%.2f")
    Machine_Utilization = st.number_input("Machine Utilization", value=safe_num("Machine_Utilization", 0.5), step=0.01, format="%.2f")
    sample_date = st.date_input("Sample date", value=vals.get("date", date.today()))
    hour_val = st.slider("Hour of Day (0-23)", 0, 23, int(vals.get("hour", 9)))

    Shift = st.selectbox("Shift", options=["Day", "Evening", "Night"], index=["Day", "Evening", "Night"].index(safe_cat("Shift", ["Day", "Evening", "Night"], "Day")))
    Machine_Type = st.selectbox("Machine Type", options=["Type_A", "Type_B", "Type_C"], index=["Type_A", "Type_B", "Type_C"].index(safe_cat("Machine_Type", ["Type_A", "Type_B", "Type_C"], "Type_A")))
    Material_Grade = st.selectbox("Material Grade", options=["Economy", "Standard", "Premium"], index=["Economy", "Standard", "Premium"].index(safe_cat("Material_Grade", ["Economy", "Standard", "Premium"], "Standard")))
    Day_of_Week = st.selectbox("Day of Week", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                               index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(safe_cat("Day_of_Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], "Friday")))

    predict_clicked = st.button("🔮 Predict Output", key="predict_btn")

with right:
    st.subheader("Model insights")
    try:
        inner = model.regressor_
        lin = inner.named_steps.get("model", None)
        if lin is not None and hasattr(lin, "coef_") and feature_columns:
            coefs = np.array(lin.coef_)
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
    st.markdown("<div class='card'><b>Quick tips</b><br>Use presets to demo sensitivity. Download the history after prediction.</div>", unsafe_allow_html=True)

# prediction, history & download
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

    rec = dict(raw_inputs)
    rec["predicted_parts_per_hour"] = pred_val
    rec["timestamp"] = datetime.now().isoformat()
    st.session_state.history.insert(0, rec)

    colA, colB = st.columns([3,1])
    with colA:
        st.markdown(f"<div class='big-result'>✅  Predicted Output: <b style='font-size:24px'>{pred_val:.2f} units/hour</b></div>", unsafe_allow_html=True)
        lower_warn = metadata.get("underperf_threshold", None)
        if lower_warn and pred_val < float(lower_warn):
            st.warning("⚠️ Predicted output below underperformance threshold — investigate inputs/maintenance.")
    with colB:
        st.metric("Pred (units/hr)", f"{pred_val:.2f}")

    # local contributions (approx — linear model only)
    contribs = compute_local_contributions(model, prepare_input_row(raw_inputs, feature_columns), feature_columns or [])
    if contribs:
        st.subheader("Top contributors (approx)")
        for name, val in contribs:
            st.write(f"{name}: {val:.3f}")

    df_history = pd.DataFrame(st.session_state.history)
    csv = df_history.to_csv(index=False)
    st.download_button("Download history (CSV)", csv, "predictions_history.csv", "text/csv")

# show recent history
if st.session_state.history:
    st.subheader("Recent predictions")
    dfh = pd.DataFrame(st.session_state.history)
    st.dataframe(dfh.head(10), use_container_width=True)
