# app.py - Heart Disease Detection Streamlit App
# Paste this into Heart_Disease_Project/app.py

import os
import io
import json
from pathlib import Path
import urllib.request
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt

# ---------------------------
# Config / Paths
# ---------------------------
st.set_page_config(
    page_title="Heart Disease Detection",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Resolve base directory (supports running from repo root or inside folder)
CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent  # Heart_Disease_Project folder
MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "heart_model_v1.joblib"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
COMPARISON_PATH = MODELS_DIR / "model_comparison.json"

# Environment secrets
MODEL_DOWNLOAD_URL = os.getenv("MODEL_DOWNLOAD_URL")  # set in Streamlit Secrets if model not in repo
MODEL_PATH_ENV = os.getenv("MODEL_PATH")  # optional override

if MODEL_PATH_ENV:
    MODEL_PATH = Path(MODEL_PATH_ENV)
else:
    MODEL_PATH = DEFAULT_MODEL_PATH

# Expected feature order (must match training)
FEATURE_ORDER = [
    "age","sex","chest_pain_type","resting_blood_pressure","cholesterol",
    "fasting_blood_sugar","resting_ecg","max_heart_rate","exercise_induced_angina",
    "st_depression","st_slope","num_major_vessels","thalassemia"
]

# ---------------------------
# Utilities
# ---------------------------
def ensure_model_file():
    """Ensure model exists locally. If not and MODEL_DOWNLOAD_URL is set, download it."""
    if MODEL_PATH.exists():
        return True
    if MODEL_DOWNLOAD_URL:
        st.sidebar.info("Downloading model from MODEL_DOWNLOAD_URL...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, MODEL_PATH)
            st.sidebar.success("Model downloaded.")
            return True
        except Exception as e:
            st.sidebar.error(f"Failed to download model: {e}")
            return False
    return False

@st.cache_resource(show_spinner=False)
def load_bundle(path: Path):
    """Load the joblib bundle (pipeline, feature_order, metrics, model_name)."""
    bundle = joblib.load(path)
    pipeline = bundle.get("pipeline")
    feature_order = bundle.get("feature_order", FEATURE_ORDER)
    metrics = bundle.get("metrics", {})
    model_name = bundle.get("model_name", "model")
    return pipeline, feature_order, metrics, model_name

def predict_df(pipeline, X: pd.DataFrame):
    """Return predictions and probabilities (if available)."""
    preds = pipeline.predict(X)
    proba = None
    try:
        proba = pipeline.predict_proba(X)[:, 1]
    except Exception:
        try:
            # some classifiers use decision_function; convert to pseudo-prob via sigmoid
            df = pipeline.decision_function(X)
            proba = 1/(1 + np.exp(-df))
        except Exception:
            proba = None
    return preds, proba

def download_button_df(df: pd.DataFrame, filename: str = "predictions.csv"):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions CSV", csv_bytes, file_name=filename, mime="text/csv")

# ---------------------------
# App start: ensure model then load
# ---------------------------
st.title("Heart Disease Detection")
st.markdown("Predict presence of heart disease using a trained classification model.")

with st.spinner("Checking model..."):
    model_available = ensure_model_file()

if not model_available:
    st.error(
        "Trained model not found. "
        "Place the model at: `Heart_Disease_Project/models/heart_model_v1.joblib` "
        "or set `MODEL_DOWNLOAD_URL` as a Streamlit Secret."
    )
    st.stop()

with st.spinner("Loading model..."):
    pipeline, loaded_feature_order, BUNDLE_METRICS, MODEL_NAME = load_bundle(MODEL_PATH)

# Update FEATURE_ORDER from bundle if present
FEATURE_ORDER = loaded_feature_order

# Sidebar: Model info & quick controls
with st.sidebar:
    st.header("Model Info")
    st.subheader(MODEL_NAME)
    if BUNDLE_METRICS:
        try:
            st.metric("Recall", f"{BUNDLE_METRICS.get('recall', 0):.3f}")
            st.write("Precision:", f"{BUNDLE_METRICS.get('precision', 0):.3f}")
            st.write("F1:", f"{BUNDLE_METRICS.get('f1', 0):.3f}")
            if BUNDLE_METRICS.get("roc_auc") is not None:
                st.write("ROC AUC:", f"{BUNDLE_METRICS.get('roc_auc', 0):.3f}")
        except Exception:
            st.write(BUNDLE_METRICS)
    else:
        st.write("No metrics found in bundle.")
    st.markdown("---")
    st.subheader("Controls")
    threshold = st.slider("Decision threshold (probability -> positive class)", 0.0, 1.0, 0.5, 0.01)
    st.caption("Adjust threshold to trade-off sensitivity/precision.")
    st.markdown("---")
    st.markdown("Project")
    st.write("- Dataset: heart_disease_dataset (400 rows)")
    st.write("- Features:", len(FEATURE_ORDER))
    st.markdown("---")
    if Path(MANIFEST_PATH).exists():
        try:
            manifest = json.loads(Path(MANIFEST_PATH).read_text())
            st.write("Manifest")
            st.write(manifest)
        except Exception:
            pass

# ---------------------------
# Main layout: two columns
# ---------------------------
col1, col2 = st.columns([2, 1])

# ---------------------------
# Single prediction form
# ---------------------------
with col1:
    st.header("Single patient prediction")

    def single_input_defaults():
        # return a dict mapping features to default values for UI
        return {
            "age": 55,
            "sex": 1,
            "chest_pain_type": 1,
            "resting_blood_pressure": 130,
            "cholesterol": 230,
            "fasting_blood_sugar": 0,
            "resting_ecg": 0,
            "max_heart_rate": 150,
            "exercise_induced_angina": 0,
            "st_depression": 0.5,
            "st_slope": 1,
            "num_major_vessels": 0,
            "thalassemia": 2
        }

    defaults = single_input_defaults()
    with st.form("single_form"):
        # create compact grid
        r1 = st.columns(4)
        age = r1[0].number_input("Age", min_value=1, max_value=120, value=defaults["age"])
        sex = r1[1].selectbox("Sex", options=[0, 1], index=1, format_func=lambda x: "Male" if x == 1 else "Female")
        cp = r1[2].selectbox("Chest pain type", options=[0, 1, 2, 3], index=defaults["chest_pain_type"])
        trestbps = r1[3].number_input("Resting BP", min_value=50, max_value=250, value=defaults["resting_blood_pressure"])

        r2 = st.columns(4)
        chol = r2[0].number_input("Cholesterol", min_value=50, max_value=600, value=defaults["cholesterol"])
        fbs = r2[1].selectbox("Fasting blood sugar >120 mg/dl", options=[0, 1], index=defaults["fasting_blood_sugar"])
        restecg = r2[2].selectbox("Resting ECG", options=[0, 1, 2], index=defaults["resting_ecg"])
        thalach = r2[3].number_input("Max heart rate", min_value=50, max_value=250, value=defaults["max_heart_rate"])

        r3 = st.columns(4)
        exang = r3[0].selectbox("Exercise-induced angina", options=[0, 1], index=defaults["exercise_induced_angina"])
        oldpeak = r3[1].number_input("ST depression (oldpeak)", value=float(defaults["st_depression"]), format="%.2f")
        slope = r3[2].selectbox("ST slope", options=[0, 1, 2], index=defaults["st_slope"])
        ca = r3[3].selectbox("Num major vessels (0-3)", options=[0, 1, 2, 3], index=defaults["num_major_vessels"])

        thal = st.selectbox("Thalassemia (1=normal,2=fixed,3=reversible)", options=[1, 2, 3], index=1)

        submit = st.form_submit_button("Predict")

    if submit:
        # create DataFrame with correct feature order
        sample = pd.DataFrame([[
            age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        ]], columns=FEATURE_ORDER)

        try:
            preds, proba = predict_df(pipeline, sample)
            pred_label = preds[0]
            prob_val = None if proba is None else float(proba[0])
            if prob_val is not None:
                predicted = int(prob_val >= threshold)
                st.success(f"Predicted: {'Heart Disease' if predicted == 1 else 'No Heart Disease'}")
                st.write(f"Model probability of heart disease: **{prob_val:.3f}** (threshold: {threshold:.2f})")
            else:
                st.success(f"Predicted: {'Heart Disease' if pred_label == 1 else 'No Heart Disease'}")
                st.info("Model did not provide probabilities.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------------------
# Batch prediction
# ---------------------------
with col2:
    st.header("Batch prediction")
    st.write("Upload a CSV with columns (any order):")
    st.write(", ".join(FEATURE_ORDER))
    uploaded = st.file_uploader("Upload CSV file for batch prediction", type=["csv"], key="batch")

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None

        if df is not None:
            missing = [c for c in FEATURE_ORDER if c not in df.columns]
            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            else:
                X = df[FEATURE_ORDER].copy()
                with st.spinner("Running predictions..."):
                    preds, proba = predict_df(pipeline, X)
                df["prediction"] = preds
                if proba is not None:
                    df["probability"] = proba
                st.write(df.head(10))
                download_button_df(df, filename="heart_predictions.csv")

# ---------------------------
# Evaluation & metrics (wide area)
# ---------------------------
st.markdown("---")
st.header("Evaluation (upload labeled test set)")

eval_col1, eval_col2 = st.columns([1, 2])

with eval_col1:
    eval_upload = st.file_uploader("Upload labeled CSV with 'heart_disease' column", type=["csv"], key="eval_upload")
    if eval_upload:
        eval_df = pd.read_csv(eval_upload)
        required = FEATURE_ORDER + ["heart_disease"]
        missing = [c for c in required if c not in eval_df.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
            eval_df = None

with eval_col2:
    if 'eval_df' in locals() and eval_df is not None:
        X_test = eval_df[FEATURE_ORDER]
        y_test = eval_df["heart_disease"]
        with st.spinner("Computing evaluation metrics..."):
            y_pred = pipeline.predict(X_test)
            try:
                y_proba = pipeline.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None

            rec = recall_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

            st.subheader("Metrics (on uploaded test set)")
            st.write(f"Recall: **{rec:.3f}**, Precision: **{prec:.3f}**, F1: **{f1:.3f}**, Accuracy: **{acc:.3f}**")
            if roc_auc is not None:
                st.write(f"ROC AUC: **{roc_auc:.3f}**")

            st.text("Classification report:")
            st.text(classification_report(y_test, y_pred, digits=4))

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4,4))
            ax.imshow(cm, cmap="Blues")
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            for (i, j), z in np.ndenumerate(cm):
                ax.text(j, i, str(z), ha='center', va='center', color='black', fontsize=12)
            st.pyplot(fig)

            # ROC curve
            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                ax2.plot([0,1], [0,1], "--", color="gray")
                ax2.set_xlabel("FPR")
                ax2.set_ylabel("TPR")
                ax2.legend()
                st.pyplot(fig2)

# ---------------------------
# Model comparison and feature importances
# ---------------------------
st.markdown("---")
st.header("Model comparison & feature importances")

# show model comparison if available
if Path(COMPARISON_PATH).exists():
    try:
        comp = json.loads(Path(COMPARISON_PATH).read_text())
        comp_df = pd.DataFrame(comp).T
        if "recall" in comp_df.columns:
            comp_df = comp_df[["recall","precision","f1","roc_auc"]]
        st.subheader("Model comparison (saved results)")
        st.dataframe(comp_df)
    except Exception:
        st.info("Could not parse model_comparison.json")

# feature importances (simple)
st.subheader("Feature importance")
try:
    clf = pipeline.named_steps.get("clf") if hasattr(pipeline, "named_steps") else None
    importances = None
    if clf is not None:
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            coef = np.array(clf.coef_)
            if coef.ndim == 2:
                coef = coef[0]
            importances = np.abs(coef)
    if importances is not None:
        imp_df = pd.DataFrame({"feature": FEATURE_ORDER, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).set_index("feature")
        st.bar_chart(imp_df)
    else:
        st.info("The model does not expose feature importances. Consider permutation importance or SHAP for richer explanations.")
except Exception as e:
    st.warning(f"Could not compute importances: {e}")

st.markdown("---")
st.caption("App built for demonstration. For reproducibility, refer to the code repo and model manifest.")
