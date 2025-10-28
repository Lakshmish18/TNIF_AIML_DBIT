# app.py
from preprocessing_helpers import QuantileClipper  # must be available before unpickling
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from typing import Optional

MODEL_PATH = "artifacts/final_ridge_pipeline.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run notebook save step first.")

# Load model once at startup
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Manufacturing Output Prediction API", version="1.0")

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
    Temperature_Pressure_Ratio: Optional[float] = None
    Total_Cycle_Time: Optional[float] = None
    Efficiency_Score: Optional[float] = None
    Machine_Utilization: Optional[float] = None

@app.get("/")
def root():
    return {"message": "Manufacturing Output Prediction API", "model_file": os.path.basename(MODEL_PATH)}

@app.post("/predict/")
def predict(payload: PredictRequest):
    try:
        data = payload.dict()
        if data.get("Temperature_Pressure_Ratio") is None:
            data["Temperature_Pressure_Ratio"] = data["Injection_Temperature"] / (data["Injection_Pressure"] + 1e-6)
        if data.get("Total_Cycle_Time") is None:
            data["Total_Cycle_Time"] = data["Cycle_Time"] + data["Cooling_Time"]

        df = pd.DataFrame([data])
        pred = model.predict(df)
        return {"Predicted_Parts_Per_Hour": float(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
