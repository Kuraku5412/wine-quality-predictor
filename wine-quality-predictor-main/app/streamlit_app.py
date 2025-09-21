# === streamlit_app.py ===
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # app/
MODEL_PATH = BASE_DIR.parent / "models" / "wine_quality_model.joblib"
FEATURES_PATH = BASE_DIR.parent / "models" / "feature_names.joblib"

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

model, features = load_model()

st.set_page_config(page_title="Wine Quality Predictor", layout="centered")

st.title("Boutique Winery — Red Wine Quality Predictor")
st.write("Enter the chemical attributes of a wine sample to predict if it's 'Good' (quality ≥ 7).")

# Build inputs dynamically
user_inputs = {}
with st.form("input_form"):
    for feat in features:
        # float inputs: adapt min/max from typical wine ranges or leave wide
        val = st.number_input(f"{feat}", format="%.6f", value=0.0, help=f"Enter {feat}")
        user_inputs[feat] = val
    submitted = st.form_submit_button("Predict")

if submitted:
    X = pd.DataFrame([user_inputs], columns=features)
    # predict_proba; model should have predict_proba after calibration
    proba = model.predict_proba(X)[0, 1]
    pred = int(model.predict(X)[0])
    label = "Good (quality ≥ 7)" if pred == 1 else "Not good (< 7)"
    # display results
    st.subheader("Prediction")
    st.write(f"**Result:** {label}")
    st.write(f"**Confidence (probability of Good):** {proba:.3f}")

    # optional: show decision threshold and explain
    thresh = 0.5
    st.write(f"Decision threshold = {thresh}. You can change this in the app code to reflect higher selectivity.")
    # give human-friendly guidance
    if proba > 0.8:
        st.success("High confidence — this sample is likely premium.")
    elif proba > 0.6:
        st.info("Moderate confidence — borderline. Consider additional testing.")
    else:
        st.warning("Low probability — sample is unlikely to meet premium standard.")

