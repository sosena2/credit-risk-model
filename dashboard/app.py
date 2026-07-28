import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bati Bank Credit Risk Explorer", layout="wide")
st.title("Bati Bank — Credit Risk Scoring Dashboard")
st.caption("Interactive tool for loan officers to score applicants and see why.")

@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")  # adjust to your saved model path

model = load_model()

st.sidebar.header("Applicant Features")
recency = st.sidebar.slider("Recency (days since last transaction)", 0, 365, 30)
frequency = st.sidebar.slider("Frequency (transaction count)", 1, 100, 10)
monetary = st.sidebar.number_input("Monetary (total transaction value)", 0.0, 100000.0, 500.0)
avg_txn = st.sidebar.number_input("Average transaction amount", 0.0, 10000.0, 50.0)
product_category = st.sidebar.selectbox("Product category", ["A", "B", "C"])
channel = st.sidebar.selectbox("Channel", ["web", "app", "ussd"])

input_df = pd.DataFrame([{
    "recency": recency, "frequency": frequency, "monetary": monetary,
    "avg_transaction_amount": avg_txn,
    "product_category": product_category, "channel_id": channel,
}])

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Risk Score")
    proba = model.predict_proba(input_df)[0][1]
    st.metric("Predicted default risk", f"{proba:.1%}")
    risk_label = "🔴 High Risk" if proba >= 0.5 else "🟢 Low Risk"
    st.markdown(f"### {risk_label}")

with col2:
    st.subheader("Why this score? (SHAP)")
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

st.divider()
st.subheader("Global Feature Importance")
st.info("Loads a precomputed SHAP summary from the training set (see notebooks/shap_analysis.ipynb).")