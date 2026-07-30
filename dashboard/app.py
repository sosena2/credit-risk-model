# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt
import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri("file:./mlruns")

st.set_page_config(page_title="Bati Bank Credit Risk Explorer", layout="wide")
st.title("Bati Bank — Credit Risk Scoring Dashboard")
st.caption("Interactive tool for loan officers to score applicants and see why.")

@st.cache_resource
def load_model():
    return mlflow.sklearn.load_model("models:/credit_risk_best_model/latest")

model = load_model()

FEATURE_COLUMNS = [
    "Amount", "Value", "PricingStrategy",
    "transaction_hour", "transaction_day", "transaction_month", "transaction_year"
]

st.sidebar.header("Transaction Details")
amount = st.sidebar.number_input("Amount", value=1000.0)
value = st.sidebar.number_input("Value", value=1000.0)
pricing_strategy = st.sidebar.selectbox("Pricing Strategy", [0, 1, 2, 4])
transaction_hour = st.sidebar.slider("Transaction Hour", 0, 23, 12)
transaction_day = st.sidebar.slider("Transaction Day", 1, 31, 15)
transaction_month = st.sidebar.slider("Transaction Month", 1, 12, 6)
transaction_year = st.sidebar.number_input("Transaction Year", value=2026, step=1)

input_df = pd.DataFrame([{
    "Amount": amount,
    "Value": value,
    "PricingStrategy": pricing_strategy,
    "transaction_hour": transaction_hour,
    "transaction_day": transaction_day,
    "transaction_month": transaction_month,
    "transaction_year": transaction_year,
}])[FEATURE_COLUMNS]

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Risk Score")
    proba = model.predict_proba(input_df)[0][1]
    st.metric("Predicted default risk", f"{proba:.1%}")
    risk_label = "🔴 High Risk" if proba >= 0.5 else "🟢 Low Risk"
    st.markdown(f"### {risk_label}")
    st.caption("Note: due to class imbalance, this model's recommended decision "
               "threshold may differ from 0.5 — see presentation for details.")

with col2:
    st.subheader("Why this score? (SHAP)")
    try:
        explainer = shap.Explainer(model.named_steps["clf"], feature_names=FEATURE_COLUMNS)
        transformed_input = model.named_steps["imputer"].transform(input_df)
        if "scaler" in model.named_steps:
            transformed_input = model.named_steps["scaler"].transform(transformed_input)
        shap_values = explainer(transformed_input)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP explanation unavailable for this model type: {e}")

st.divider()
st.subheader("About this model")
st.write(
    "This model predicts a proxy risk label built from customer transaction behavior "
    "(recency, frequency, monetary value), since traditional credit history isn't available. "
    "Test set performance: ROC-AUC ≈ 0.84."
)