# tests/test_smoke.py
# Quick fix: ensure QuantileClipper is available in the namespace before joblib.load
# This imports the module that defines the custom transformer so pickle can find it.
import importlib
# import the helper module that contains QuantileClipper
import preprocessing_helpers
import joblib
import pandas as pd

def test_model_prediction():
    """
    Smoke test to verify that:
    1. The saved Ridge model loads correctly (ensuring QuantileClipper is imported first).
    2. It can make a prediction on a valid single-row input.
    """
    # ensure module is loaded (redundant but explicit)
    importlib.reload(preprocessing_helpers)

    model_path = "artifacts/final_ridge_pipeline.joblib"

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise AssertionError(f"Model failed to load: {e}")

    sample = pd.DataFrame([{
        "Injection_Temperature": 220,
        "Injection_Pressure": 120,
        "Cycle_Time": 30,
        "Cooling_Time": 12,
        "Material_Viscosity": 350,
        "Ambient_Temperature": 27,
        "Machine_Age": 5,
        "Operator_Experience": 8,
        "Maintenance_Hours": 60,
        "Shift": "Day",
        "Machine_Type": "Type_A",
        "Material_Grade": "Standard",
        "Day_of_Week": "Monday",
        "Efficiency_Score": 0.07,
        "Machine_Utilization": 0.55
    }])

    try:
        preds = model.predict(sample)
    except Exception as e:
        raise AssertionError(f"Prediction failed: {e}")

    assert len(preds) == 1
    assert isinstance(preds[0], (float, int))
    print("✅ Smoke test passed! Predicted Parts_Per_Hour:", preds[0])

if __name__ == "__main__":
    test_model_prediction()
