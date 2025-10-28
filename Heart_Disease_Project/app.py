# app.py
import os
from dotenv import load_dotenv
import streamlit as st
import joblib, json, tempfile, requests
import pandas as pd, numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# Load .env from Desktop (outside repo) by default
dotenv_path = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", ".env")
load_dotenv(dotenv_path=dotenv_path)

MODEL_PATH = os.getenv("MODEL_PATH", "models/heart_model_v1.joblib")
MODEL_DOWNLOAD_URL = os.getenv("MODEL_DOWNLOAD_URL", None)

st.set_page_config(page_title="Heart Disease Detection", layout="wide")

@st.cache_resource
def load_bundle():
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
    elif MODEL_DOWNLOAD_URL:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".joblib")
        r = requests.get(MODEL_DOWNLOAD_URL, stream=True, timeout=60)
        r.raise_for_status()
        with open(tmp.name, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        bundle = joblib.load(tmp.name)
    else:
        st.error("Model not found. Add models/heart_model_v1.joblib or set MODEL_DOWNLOAD_URL in .env on Desktop.")
        st.stop()
    return bundle

bundle = load_bundle()
pipeline = bundle["pipeline"]
FEATURE_ORDER = bundle.get("feature_order", [])
METRICS = bundle.get("metrics", {})
MODEL_NAME = bundle.get("model_name", "model")

st.title("❤️ Heart Disease Detection")
st.sidebar.title("Model & Controls")
st.sidebar.write(f"Deployed model: **{MODEL_NAME}**")
st.sidebar.subheader("Saved metrics (manifest)")
st.sidebar.json(METRICS)

# Show model comparison if present
comparison_path = os.path.join("models", "model_comparison.json")
if os.path.exists(comparison_path):
    with open(comparison_path) as f:
        comp = json.load(f)
    st.sidebar.subheader("Model Comparison")
    st.sidebar.dataframe(pd.DataFrame(comp).T)

mode = st.radio("Mode", ["Single prediction", "Batch CSV", "Evaluation", "Model Info"])

if mode == "Single prediction":
    st.header("Single patient prediction")
    if not FEATURE_ORDER:
        st.error("Model does not include feature order. Add feature_order when saving model.")
    else:
        inputs = {}
        cols = st.columns(2)
        for i, feat in enumerate(FEATURE_ORDER):
            with cols[i % 2]:
                if feat in ["age", "resting_blood_pressure", "cholesterol", "max_heart_rate", "num_major_vessels"]:
                    val = st.number_input(feat, value=0, step=1)
                else:
                    val = st.number_input(feat, value=0.0, format="%.3f")
                inputs[feat] = val
        if st.button("Predict"):
            X = np.array([list(inputs.values())])
            pred = int(pipeline.predict(X)[0])
            proba = pipeline.predict_proba(X)[0][1] if hasattr(pipeline, "predict_proba") else None
            st.success(f"Prediction: **{'Heart Disease (1)' if pred==1 else 'No Heart Disease (0)'}**")
            if proba is not None:
                st.write(f"Probability (class=1): {proba:.4f}")
                st.progress(min(100, int(proba * 100)))

elif mode == "Batch CSV":
    st.header("Batch prediction (upload CSV)")
    st.markdown("CSV should have the same columns as the feature order in the sidebar (or columns in the same order).")
    if FEATURE_ORDER:
        st.info("Feature order: " + ", ".join(FEATURE_ORDER))
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        missing = [c for c in FEATURE_ORDER if c not in df.columns]
        if missing and df.shape[1] == len(FEATURE_ORDER):
            df.columns = FEATURE_ORDER
            st.info("Assigned columns by position.")
        elif missing:
            st.error(f"CSV missing columns: {missing}")
            st.stop()
        df_in = df[FEATURE_ORDER]
        if st.button("Run batch prediction"):
            preds = pipeline.predict(df_in)
            df_out = df.copy()
            df_out["prediction"] = preds
            if hasattr(pipeline, "predict_proba"):
                df_out["probability"] = pipeline.predict_proba(df_in)[:, 1]
            st.dataframe(df_out.head(100))
            csv = df_out.to_csv(index=False).encode()
            st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")

elif mode == "Evaluation":
    st.header("Evaluation (upload labeled CSV with 'heart_disease')")
    uploaded = st.file_uploader("Upload labeled CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if "heart_disease" not in df.columns:
            st.error("Uploaded CSV must contain 'heart_disease' column.")
        else:
            X = df[FEATURE_ORDER]
            y = df["heart_disease"]
            y_pred = pipeline.predict(X)
            probs = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, "predict_proba") else None
            st.subheader("Classification Report")
            st.text(classification_report(y, y_pred))
            if probs is not None:
                st.write("ROC-AUC:", roc_auc_score(y, probs))
            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots()
            ax.matshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, f"{v}", ha="center", va="center")
            st.pyplot(fig)

else:  # Model Info
    st.header("Model info & explainability")
    st.subheader("Saved metrics")
    st.json(METRICS)
    try:
        model_obj = pipeline.named_steps.get("clf", pipeline)
        if hasattr(model_obj, "feature_importances_"):
            importances = model_obj.feature_importances_
            fi = pd.Series(importances, index=FEATURE_ORDER).sort_values(ascending=False)
            st.subheader("Top feature importances")
            st.dataframe(fi.head(10))
            fig, ax = plt.subplots(figsize=(6,4))
            fi.plot.bar(ax=ax)
            st.pyplot(fig)
        elif hasattr(model_obj, "coef_"):
            coef = model_obj.coef_
            if coef.ndim == 1:
                s = pd.Series(coef, index=FEATURE_ORDER).sort_values(key=abs, ascending=False)
                st.subheader("Top coefficients")
                st.dataframe(s.head(10))
    except Exception as e:
        st.write("Explainability not available:", e)
