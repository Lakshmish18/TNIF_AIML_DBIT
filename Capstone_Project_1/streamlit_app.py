# Place near top of streamlit_app.py (imports)
import os
import json
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from datetime import datetime, date, time

# ---------- helper: load feature list saved when you trained the model ----------
BASE_DIR = os.path.dirname(__file__)
FEATURES_JSON_PATH = os.path.join(BASE_DIR, "artifacts", "feature_columns.json")  # adjust if you saved elsewhere

def load_feature_columns(path=FEATURES_JSON_PATH):
    if os.path.exists(path):
        with open(path, "r") as f:
            cols = json.load(f)
        return cols
    # fallback: return None
    return None

feature_columns = load_feature_columns()

# ---------- helper: prepare input row exactly as training pipeline expected ----------
def prepare_input_row(inputs):
    """
    inputs: dict of raw UI inputs (must contain Injection_Temperature, Injection_Pressure, Cycle_Time, Cooling_Time, etc.)
    returns: pd.DataFrame with one row and columns matching feature_columns (fills missing with 0 or median-like defaults)
    """
    # copy to avoid mutating caller dict
    d = dict(inputs)

    # 1) derived features we expect (adjust formulas to match your training code)
    # Total cycle time
    d["Total_Cycle_Time"] = float(d.get("Cycle_Time", 0)) + float(d.get("Cooling_Time", 0))

    # Cycle/Cooling ratio (avoid division by zero)
    cooling = float(d.get("Cooling_Time", 0))
    cycle = float(d.get("Cycle_Time", 0))
    d["Cycle_Cooling_Ratio"] = float(cycle) / (cooling + 1e-6)

    # Temperature x Pressure (some pipelines use product; adapt if your pipeline used ratio instead)
    d["Temp_Pressure_Product"] = float(d.get("Injection_Temperature", 0)) * float(d.get("Injection_Pressure", 0))

    # Temperature / Pressure ratio (if used)
    p = float(d.get("Injection_Pressure", 1))
    d["Temperature_Pressure_Ratio"] = float(d.get("Injection_Temperature", 0)) / (p + 1e-6)

    # 2) time/date related features (ensure UI provides `sample_date` and `sample_time` or a datetime)
    # If UI contains 'datetime' or separate date/time fields, use them. Otherwise default to now.
    dt = d.get("datetime_obj")  # allow passing an actual datetime object
    if dt is None:
        # if app provided 'date' and 'hour' separately, you can construct datetime:
        # e.g. date_val = d.get("date", date.today()); hour_val = int(d.get("hour", 0))
        date_val = d.get("date", date.today())
        hour_val = int(d.get("hour", 0))
        try:
            if isinstance(date_val, str):
                date_val = datetime.fromisoformat(date_val).date()
            dt = datetime.combine(date_val, time(hour=hour_val))
        except Exception:
            dt = datetime.now()

    d["Hour"] = dt.hour
    d["Day"] = dt.day
    d["Month"] = dt.month
    d["Year"] = dt.year
    # DayOfWeek_num: monday=0 .. sunday=6 — adjust +1 if your training used 1..7
    d["DayOfWeek_num"] = dt.weekday()

    # 3) ensure categorical one-hot placeholders exist if pipeline expects names
    # (We will rely on pipeline's ColumnTransformer + OneHotEncoder. It expects raw categorical columns e.g. 'Shift','Machine_Type','Material_Grade','Day_of_Week')
    # So keep them as present in inputs
    # d["Shift"] = d.get("Shift","Day")  # already present from UI

    # 4) Build DataFrame with one row
    df = pd.DataFrame([d])

    # 5) Ensure all required columns in feature_columns exist in df — if not, add with zeros
    if feature_columns is not None:
        for c in feature_columns:
            if c not in df.columns:
                # choose a reasonable default: 0 for numeric, "" for object
                df[c] = 0

        # reorder to exactly the feature_columns order
        df = df[feature_columns]

    # final type casting: convert numeric-like columns to numeric
    for col in df.columns:
        # try to convert to numeric if possible (silently ignore exceptions)
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            pass

    return df

# ---------- Example integration with Streamlit UI ----------
# Suppose earlier you used st.number_input, st.selectbox, etc. Gather them into a dict:

# Example UI (use your existing UI values)
injection_temp = st.number_input("Injection Temperature", value=220.0, step=0.1)
injection_pressure = st.number_input("Injection Pressure", value=130.0, step=0.1)
cycle_time = st.number_input("Cycle Time", value=30.0, step=0.1)
cooling_time = st.number_input("Cooling Time", value=12.0, step=0.1)
material_viscosity = st.number_input("Material Viscosity", value=300.0, step=0.1)
ambient_temp = st.number_input("Ambient Temperature", value=25.0, step=0.1)

# date/time picker for Hour / Day
sample_date = st.date_input("Sample date", value=date.today())
sample_hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=9)

# categorical picks
shift = st.selectbox("Shift", ["Day", "Night", "Evening"])
machine_type = st.selectbox("Machine Type", ["Type_A", "Type_B"])
material_grade = st.selectbox("Material Grade", ["Economy", "Standard", "Premium"])
day_of_week = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

# other fields you use
machine_age = st.number_input("Machine Age (years)", value=3.0)
operator_experience = st.number_input("Operator Experience (years)", value=5.0)
maintenance_hours = st.number_input("Maintenance Hours (per month)", value=20.0)
efficiency_score = st.number_input("Efficiency Score", value=0.05)
machine_utilization = st.number_input("Machine Utilization", value=0.5)

# Build raw input dict the same way as training notebook
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

# ---------- Predict button ----------
if st.button("Predict Output"):
    try:
        # prepare dataframe with all columns model expects
        X_pred = prepare_input_row(raw_inputs)

        # load model (ensure model loaded earlier as `model`, or load here)
        MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "final_ridge_pipeline.joblib")
        model = joblib.load(MODEL_PATH)

        # predict: model expects same columns/order as feature_columns, returns array-like
        y_hat = model.predict(X_pred)
        st.success(f"Predicted Parts per hour: {float(y_hat[0]):.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        # also log df columns for debugging
        try:
            st.write("Prepared input columns:", list(X_pred.columns))
        except Exception:
            pass
