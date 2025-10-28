# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import os
from preprocessing_helpers import QuantileClipper  # required for loading model

# --------------------------------------------
# Streamlit page configuration
# --------------------------------------------
st.set_page_config(
    page_title="Manufacturing Output Predictor",
    page_icon="🏭",
    layout="centered",
)

# --------------------------------------------
# Load saved model
# --------------------------------------------
MODEL_PATH = "artifacts/final_ridge_pipeline.joblib"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at: {MODEL_PATH}")
        st.stop()
    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

# --------------------------------------------
# App Header
# --------------------------------------------
st.title("🏭 Manufacturing Equipment Output Prediction")
st.markdown("""
This tool predicts the **hourly output (Parts Per Hour)** of an injection molding machine  
based on its operating conditions and parameters.
""")

# --------------------------------------------
# Input Form
# --------------------------------------------
with st.form("input_form"):
    st.subheader("🧩 Machine Parameters")

    col1, col2 = st.columns(2)

    with col1:
        Injection_Temperature = st.number_input("Injection Temperature (°C)", 150.0, 300.0, 220.0)
        Cycle_Time = st.number_input("Cycle Time (seconds)", 10.0, 60.0, 30.0)
        Material_Viscosity = st.number_input("Material Viscosity", 100.0, 500.0, 350.0)
        Machine_Age = st.number_input("Machine Age (years)", 0.0, 20.0, 5.0)
        Maintenance_Hours = st.number_input("Maintenance Hours (per month)", 0.0, 100.0, 60.0)
        Machine_Type = st.selectbox("Machine Type", ["Type_A", "Type_B"])
        Material_Grade = st.selectbox("Material Grade", ["Economy", "Standard", "Premium"])

    with col2:
        Injection_Pressure = st.number_input("Injection Pressure (MPa)", 80.0, 180.0, 120.0)
        Cooling_Time = st.number_input("Cooling Time (seconds)", 5.0, 30.0, 12.0)
        Ambient_Temperature = st.number_input("Ambient Temperature (°C)", 10.0, 45.0, 27.0)
        Operator_Experience = st.number_input("Operator Experience (years)", 0.0, 25.0, 8.0)
        Shift = st.selectbox("Shift", ["Day", "Evening", "Night"])
        Day_of_Week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        Efficiency_Score = st.number_input("Efficiency Score", 0.0, 0.2, 0.07)
        Machine_Utilization = st.number_input("Machine Utilization", 0.0, 1.0, 0.55)

    submitted = st.form_submit_button("🔮 Predict Output")

# --------------------------------------------
# Prediction logic
# --------------------------------------------
if submitted:
    # Derived columns
    Temperature_Pressure_Ratio = Injection_Temperature / (Injection_Pressure + 1e-6)
    Total_Cycle_Time = Cycle_Time + Cooling_Time

    # Build dataframe for model input
    input_data = pd.DataFrame([{
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
        "Temperature_Pressure_Ratio": Temperature_Pressure_Ratio,
        "Total_Cycle_Time": Total_Cycle_Time,
        "Efficiency_Score": Efficiency_Score,
        "Machine_Utilization": Machine_Utilization
    }])

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"✅ Predicted Parts Per Hour: **{prediction:.2f}**")
        st.balloons()

        # Optional: show inputs below
        with st.expander("🔍 See your input data"):
            st.dataframe(input_data)
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
