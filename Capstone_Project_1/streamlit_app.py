"""
Streamlit app for Manufacturing Parts-per-Hour prediction.

Place this file at: Capstone_Project_1/streamlit_app.py
Ensure artifacts/final_ridge_pipeline.joblib (model) and artifacts/feature_columns.json (feature list) are present
(or can be produced).
"""

import os
import sys
import json
from datetime import datetime, date, time
from typing import Optional

import pandas as pd
import numpy as np
import joblib
import streamlit as st

# ---------- robust import so QuantileClipper is available for unpickling ----------
try:
    from Capstone_Project_1.preprocessing_helpers import QuantileClipper
except Exception:
    try:
        # when running from inside the folder
        from preprocessing_helpers import QuantileClipper
    except Exception:
        # ensure folder is on sys.path then import
        base = os.path.dirname(__file__)
        if base not in sys.path:
            sys.path.insert(0, base)
        from preprocessing_helpers import QuantileClipper

# ---------- paths ----------
BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "final_ridge_pipeline.joblib")
FEATURES_JSON_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.json")

st.set_page_config(page_title="Manufacturing Output Predictor", layout="wide")

st.title("🏭 Manufacturing Output Prediction (Parts / hour)")
st.caption("Provide machine and process parameters — app returns predicted units/hour using saved Ridge pipeline.")

# ---------- load model (safe) ----------
model = None
feature_columns = None
model_load_error = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        model_load_error = str(e)
else:
    model_load_error = f"Model not found at {MODEL_PATH}."

# If feature list not present, attempt to extract from model object after loading
if os.path.exists(FEATURES_JSON_PATH):
    try:
        with open(FEATURES_JSON_PATH, "r") as f:
            feature_columns = json.load(f)
    except Exception:
        feature_columns = None
else:
    # try to extract from model pipeline if model loaded (best-effort)
    try:
        if model is not None:
            # If model is a TransformedTargetRegressor with inner pipeline
            if hasattr(model, "regressor_"):
                inner = getattr(model, "regressor_", None)
            else:
                inner = model
            # try to find preprocess ColumnTransformer
            preprocess = None
            if hasattr(inner, "named_steps") and "preprocess" in inner.named_steps:
                preprocess = inner.named_steps["preprocess"]
            if preprocess is not None and hasattr(preprocess, "transformers_"):
                # attempt to build feature list from fitted preprocess (numerics + onehot output)
                try:
                    # If sklearn >= 1.0, ColumnTransformer has get_feature_names_out for pipelines
                    from sklearn.compose import ColumnTransformer
                    feat_names = []
                    for name, trans, cols in preprocess.transformers_:
                        if name == "num":
                            # numeric pipeline: poly may expand names
                            poly = trans.named_steps.get("poly", None) if hasattr(trans, "named_steps") else None
                            if poly is not None and hasattr(poly, "get_feature_names_out"):
                                feat_names += poly.get_feature_names_out(cols).tolist()
                            else:
                                feat_names += list(cols)
                        elif name == "cat":
                            onehot = trans.named_steps.get("onehot", None) if hasattr(trans, "named_steps") else None
                            if onehot is not None and hasattr(onehot, "get_feature_names_out"):
                                feat_names += onehot.get_feature_names_out(cols).tolist()
                            else:
                                feat_names += list(cols)
                    if feat_names:
                        feature_columns = feat_names
                except Exception:
                    feature_columns = None
    except Exception:
        feature_columns = None

# ---------- input UI ----------
st.header("Input parameters")

col1, col2, col3 = st.columns(3)

with col1:
    injection_temp = st.number_input("Injection Temperature", value=220.0, step=0.1)
    cycle_time = st.number_input("Cycle Time", value=30.0, step=0.1)
    material_viscosity = st.number_input("Material Viscosity", value=300.0, step=0.1)
    machine_age = st.number_input("Machine Age (years)", value=3.0, step=0.1)
    machine_type = st.selectbox("Machine Type", ["Type_A", "Type_B", "Type_C"])
    material_grade = st.selectbox("Material Grade", ["Economy", "Standard", "Premium"])

with col2:
    injection_pressure = st.number_input("Injection Pressure", value=130.0, step=0.1)
    cooling_time = st.number_input("Cooling Time", value=12.0, step=0.1)
    ambient_temp = st.number_input("Ambient Temperature", value=25.0, step=0.1)
    operator_experience = st.number_input("Operator Experience (years)", value=5.0, step=0.1)
    shift = st.selectbox("Shift", ["Day", "Evening", "Night"])
    day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

with col3:
    maintenance_hours = st.number_input("Maintenance Hours (per month)", value=20.0, step=0.1)
    efficiency_score = st.number_input("Efficiency Score", value=0.05, step=0.01)
    machine_utilization = st.number_input("Machine Utilization", value=0.5, step=0.01)
    sample_date = st.date_input("Sample date", value=date.today())
    sample_hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=9)

# Build raw inputs dict
raw_inputs = {
    "Injection_Temperature": injection_temp,
    "Injection_Pressure": injection_pressure,
    "Cycle_Time": cycle_time,
    "Cooling_Time": cooling_time,
    "Material_Viscosity": material_viscosity,
    "Ambient_Temperature": ambient_temp,
    "Machine_Age": machine_age,
    "Operator_Experience": operator_experience,
    "Maintenance_Hours": maintenance_hours,
    "Shift": shift,
    "Machine_Type": machine_type,
    "Material_Grade": material_grade,
    "Day_of_Week": day_of_week,
    "Efficiency_Score": efficiency_score,
    "Machine_Utilization": machine_utilization,
    "date": sample_date,
    "hour": sample_hour,
}

# ---------- helper to prepare DataFrame identical to training pipeline ----------
def prepare_input_row(inputs: dict, feature_columns_list: Optional[list] = None) -> pd.DataFrame:
    d = dict(inputs)
    # derived
    d["Total_Cycle_Time"] = float(d.get("Cycle_Time", 0)) + float(d.get("Cooling_Time", 0))
    cycle = float(d.get("Cycle_Time", 0))
    cooling = float(d.get("Cooling_Time", 0))
    d["Cycle_Cooling_Ratio"] = float(cycle) / (cooling + 1e-6)
    d["Temp_Pressure_Product"] = float(d.get("Injection_Temperature", 0)) * float(d.get("Injection_Pressure", 0))
    p = float(d.get("Injection_Pressure", 1))
    d["Temperature_Pressure_Ratio"] = float(d.get("Injection_Temperature", 0)) / (p + 1e-6)

    # time features
    try:
        date_val = d.get("date", date.today())
        hour_val = int(d.get("hour", 0))
        if isinstance(date_val, str):
            date_val = datetime.fromisoformat(date_val).date()
        dt = datetime.combine(date_val, time(hour=hour_val))
    except Exception:
        dt = datetime.now()
    d["Hour"] = dt.hour
    d["Day"] = dt.day
    d["Month"] = dt.month
    d["Year"] = dt.year
    d["DayOfWeek_num"] = dt.weekday()

    # keep categorical names (pipeline expects raw categorical columns)
    if feature_columns_list:
        # build a zero-filled row dictionary for all expected features
        row = {c: 0 for c in feature_columns_list}
        # insert computed / provided values for the basic columns (if present)
        for k, v in d.items():
            # prefer numeric conversion for numeric-like values
            row[k] = v
        df = pd.DataFrame([row], columns=feature_columns_list)
    else:
        df = pd.DataFrame([d])

    # ensure numeric conversions where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            pass

    return df

# ---------- Predict button and display ----------
st.markdown("---")
if st.button("Predict Output"):
    if model is None:
        st.error(f"Model load error: {model_load_error or 'unknown'}")
    else:
        try:
            X_pred = prepare_input_row(raw_inputs, feature_columns)
            # if feature columns exist but some categorical raw columns are missing, try to add them
            missing_cats = [c for c in ["Shift", "Machine_Type", "Material_Grade", "Day_of_Week"] if c not in X_pred.columns]
            for mc in missing_cats:
                X_pred[mc] = raw_inputs.get(mc, "")

            y_hat = model.predict(X_pred)
            st.success(f"✅ Predicted Parts per hour: {float(y_hat[0]):.2f}")
            # show prepared columns for debugging
            with st.expander("Prepared input columns (for debugging)"):
                st.write(list(X_pred.columns))
                st.write(X_pred.head(1))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            try:
                st.write("Prepared input columns:", list(X_pred.columns))
            except Exception:
                pass

st.markdown("---")
st.caption("If the app fails on unpickling, ensure preprocessing_helpers.py & __init__.py are present in the deployed repo and redeploy.")
