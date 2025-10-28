# Capstone_Project_1/streamlit_app.py
import os
import json
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from datetime import datetime, date, time

# Ensure custom transformer available for unpickling
from Capstone_Project_1.preprocessing_helpers import QuantileClipper

# Paths
BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "final_ridge_pipeline.joblib")
FEATURES_JSON_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.json")

# load feature list helper
def load_feature_columns(path=FEATURES_JSON_PATH):
    if os.path.exists(path):
        with open(path, "r") as f:
            cols = json.load(f)
        return cols
    return None

feature_columns = load_feature_columns()

# helper to construct the DataFrame expected by the pipeline
def prepare_input_row(inputs):
    d = dict(inputs)  # copy
    # derived features
    d["Total_Cycle_Time"] = float(d.get("Cycle_Time", 0)) + float(d.get("Cooling_Time", 0))
    cooling = float(d.get("Cooling_Time", 0))
    cycle = float(d.get("Cycle_Time", 0))
    d["Cycle_Cooling_Ratio"] = float(cycle) / (cooling + 1e-6)
    d["Temp_Pressure_Product"] = float(d.get("Injection_Temperature", 0)) * float(d.get("Injection_Pressure", 0))
    p = float(d.get("Injection_Pressure", 1))
    d["Temperature_Pressure_Ratio"] = float(d.get("Injection_Temperature", 0)) / (p + 1e-6)

    # ensure time features — either provided or use now / defaults
    if "date" in d and "hour" in d:
        try:
            date_val = d.get("date")
            if isinstance(date_val, str):
                date_val = datetime.fromisoformat(date_val).date()
            dt = datetime.combine(date_val, time(hour=int(d.get("hour", 0))))
        except Exception:
            dt = datetime.now()
    else:
        dt = datetime.now()
    d["Hour"] = dt.hour
    d["Day"] = dt.day
    d["Month"] = dt.month
    d["Year"] = dt.year
    d["DayOfWeek_num"] = dt.weekday()

    # Ensure raw categorical columns exist with safe defaults
    for cat_col, default in [("Shift","Day"), ("Machine_Type","Type_A"), ("Material_Grade","Standard"), ("Day_of_Week","Monday")]:
        if cat_col not in d or d.get(cat_col) is None:
            d[cat_col] = default

    # Build DataFrame
    df = pd.DataFrame([d])

    # If a list of features is available, ensure all are present and in the exact order
    if feature_columns is not None:
        for c in feature_columns:
            if c not in df.columns:
                # numeric -> 0, otherwise empty string
                df[c] = 0
        df = df[feature_columns]

    # try convert numeric-like columns to numeric
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            pass

    return df

# Streamlit UI
st.title("Manufacturing Output Prediction")
st.write("Predict parts per hour from machine operating parameters")

# Input widgets
injection_temp = st.number_input("Injection Temperature", value=220.0, step=0.1)
injection_pressure = st.number_input("Injection Pressure", value=130.0, step=0.1)
cycle_time = st.number_input("Cycle Time", value=30.0, step=0.1)
cooling_time = st.number_input("Cooling Time", value=12.0, step=0.1)
material_viscosity = st.number_input("Material Viscosity", value=300.0, step=0.1)
ambient_temp = st.number_input("Ambient Temperature", value=25.0, step=0.1)
machine_age = st.number_input("Machine Age (years)", value=3.0)
operator_experience = st.number_input("Operator Experience (years)", value=5.0)
maintenance_hours = st.number_input("Maintenance Hours (per month)", value=20.0)
efficiency_score = st.number_input("Efficiency Score", value=0.05)
machine_utilization = st.number_input("Machine Utilization", value=0.5)

shift = st.selectbox("Shift", ["Day", "Night", "Evening"])
machine_type = st.selectbox("Machine Type", ["Type_A", "Type_B", "Type_C"])
material_grade = st.selectbox("Material Grade", ["Economy", "Standard", "Premium"])
day_of_week = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

sample_date = st.date_input("Sample date", value=date.today())
sample_hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=9)

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
    "hour": sample_hour
}

if st.button("Predict Output"):
    try:
        X_pred = prepare_input_row(raw_inputs)
        # load model (preload would be more efficient; this is fine)
        model = joblib.load(MODEL_PATH)
        y_hat = model.predict(X_pred)
        st.success(f"Predicted Parts per hour: {float(y_hat[0]):.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        try:
            st.write("Prepared input columns:", list(X_pred.columns))
        except Exception:
            pass
