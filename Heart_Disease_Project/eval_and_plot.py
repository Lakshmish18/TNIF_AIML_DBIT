# eval_and_plot.py
import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = "models/heart_model_v1.joblib"
DATA_PATH = "data/heart_disease_dataset.csv"
OUT_DIR = "report"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(MODEL_PATH + " not found")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(DATA_PATH + " not found")

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
features = bundle.get("feature_order", [])

df = pd.read_csv(DATA_PATH)
X = df[features]
y = df["heart_disease"]

y_pred = pipeline.predict(X)
probs = pipeline.predict_proba(X)[:,1] if hasattr(pipeline, "predict_proba") else None

report = classification_report(y, y_pred, output_dict=True)
os.makedirs(OUT_DIR, exist_ok=True)
pd.DataFrame(report).T.to_csv(os.path.join(OUT_DIR, "classification_report_full.csv"))

cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.title("Confusion Matrix (full dataset)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_full.png"), dpi=150)

if probs is not None:
    try:
        auc = roc_auc_score(y, probs)
        print("ROC-AUC (full):", auc)
    except Exception:
        pass

print("Saved classification report and confusion matrix to report/")
