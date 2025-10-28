# train_and_save.py
"""
Train and compare classification models, save the best model (by recall) and a model_comparison.json.

Expected dataset path: data/heart_disease_dataset.csv
Outputs:
 - models/heart_model_v1.joblib  (bundle: pipeline, feature_order, metrics, model_name)
 - models/manifest.json          (metrics of chosen model)
 - models/model_comparison.json  (metrics for all models)
"""
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# Config
DATA_PATH = "data/heart_disease_dataset.csv"
MODEL_DIR = "models"
RANDOM_STATE = 42

FEATURE_COLS = [
    "age","sex","chest_pain_type","resting_blood_pressure","cholesterol",
    "fasting_blood_sugar","resting_ecg","max_heart_rate","exercise_induced_angina",
    "st_depression","st_slope","num_major_vessels","thalassemia"
]
TARGET = "heart_disease"

os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
X = df[FEATURE_COLS]
y = df[TARGET]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Define candidate models
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE)
}

results = {}
fitted_pipelines = {}

print("Training and evaluating models...")

for name, clf in candidates.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    # predict_proba may not exist for all; SVC has probability=True so it should
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred))
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    results[name] = metrics
    fitted_pipelines[name] = pipe
    print(f"\n{name} metrics: {metrics}")

# Save comparison JSON
with open(os.path.join(MODEL_DIR, "model_comparison.json"), "w") as f:
    json.dump(results, f, indent=2)

# Select best model by Recall (primary metric). Tiebreaker: F1.
def select_best(results_dict):
    best = None
    best_score = -1
    for k, v in results_dict.items():
        score = v.get("recall", 0) or 0
        if score > best_score:
            best = k
            best_score = score
        elif score == best_score:
            # tiebreaker by f1
            if v.get("f1", 0) > results_dict[best].get("f1", 0):
                best = k
    return best

best_name = select_best(results)
best_pipe = fitted_pipelines[best_name]
best_metrics = results[best_name]

# Save best pipeline as bundle
bundle = {
    "pipeline": best_pipe,
    "feature_order": FEATURE_COLS,
    "metrics": best_metrics,
    "model_name": best_name
}
joblib.dump(bundle, os.path.join(MODEL_DIR, "heart_model_v1.joblib"))

# Save manifest (chosen model metrics)
with open(os.path.join(MODEL_DIR, "manifest.json"), "w") as f:
    json.dump({"model_name": best_name, "metrics": best_metrics}, f, indent=2)

print(f"\nSaved best model: {best_name}")
print("Bundle saved to models/heart_model_v1.joblib")
print("Comparison saved to models/model_comparison.json")
