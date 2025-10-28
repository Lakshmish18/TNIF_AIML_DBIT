# app.py - simple FastAPI wrapper to serve the model as an API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
import sys

# ensure package import for unpickling
try:
    from Capstone_Project_1.preprocessing_helpers import QuantileClipper
except Exception:
    try:
        from preprocessing_helpers import QuantileClipper
    except Exception:
        base = os.path.dirname(__file__)
        if base not in sys.path:
            sys.path.insert(0, base)
        from preprocessing_helpers import QuantileClipper

MODEL_PATH = "Capstone_Project_1/artifacts/final_ridge_pipeline.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}, add or move final_ridge_pipeline.joblib to that path.")

model = joblib.load(MODEL_PATH)

app = FastAPI(title="Manufacturing Output Prediction API")

class PredictRequest(BaseModel):
    Injection_Temperature: float
    Injection_Pressure: float
    Cycle_Time: float
    Cooling_Time: float
    Material_Viscosity: float
    Ambient_Temperature: float
    Machine_Age: float
    Operator_Experience: float
    Maintenance_Hours: float
    Shift: str
    Machine_Type: str
    Material_Grade: str
    Day_of_Week: str
    date: str
    hour: int

@app.get("/")
def root():
    return {"message": "Manufacturing Output Prediction API"}

@app.post("/predict/")
def predict(req: PredictRequest):
    try:
        data = req.dict()
        # derive features similarly as streamlit app
        data["Total_Cycle_Time"] = data["Cycle_Time"] + data["Cooling_Time"]
        data["Cycle_Cooling_Ratio"] = data["Cycle_Time"] / (data["Cooling_Time"] + 1e-6)
        data["Temp_Pressure_Product"] = data["Injection_Temperature"] * data["Injection_Pressure"]
        data["Temperature_Pressure_Ratio"] = data["Injection_Temperature"] / (data["Injection_Pressure"] + 1e-6)

        # build DataFrame with one row
        X = pd.DataFrame([data])
        pred = model.predict(X)
        return {"predicted_parts_per_hour": float(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
