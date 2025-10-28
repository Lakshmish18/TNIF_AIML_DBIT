# show_importances.py
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATH = "models/heart_model_v1.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(MODEL_PATH + " not found")

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
feature_order = bundle.get("feature_order", [])

model_obj = pipeline.named_steps.get("clf", pipeline)
if not hasattr(model_obj, "feature_importances_"):
    raise AttributeError("Model has no feature_importances_. Not a tree-based model.")

importances = model_obj.feature_importances_
fi = pd.Series(importances, index=feature_order).sort_values(ascending=False)

os.makedirs("report", exist_ok=True)
fi.to_csv("report/feature_importances.csv", header=["importance"])

plt.figure(figsize=(8,6))
fi.plot.bar()
plt.title("Feature Importances (RandomForest)")
plt.tight_layout()
plt.savefig("report/feature_importances.png", dpi=150)
print("Saved report/feature_importances.csv and report/feature_importances.png")
