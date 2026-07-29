# Bati Bank Credit Risk Model

A machine learning system that scores customer creditworthiness using alternative transactional data, enabling Bati Bank to safely extend buy-now-pay-later credit to customers without traditional credit histories.

![CI](https://github.com/sosena2/credit-risk-model/actions/workflows/ci.yml/badge.svg)

## Business Problem

Bati Bank wants to offer a "buy now, pay later" service through an eCommerce partner, but has no traditional credit bureau data on most applicants. Without a reliable risk model, the bank faces a choice between over-cautious rejection (lost revenue) and under-informed approval (default losses).

## Solution Overview

Using RFM (Recency, Frequency, Monetary) analysis on transaction histories, we construct a proxy default-risk label via a percentile-rank composite score (the bottom-engagement quartile of customers is labeled high risk). Three classifiers — Logistic Regression, Random Forest, and Gradient Boosting — are trained and compared with hyperparameter search, tracked via MLflow. The best model is served through an interactive Streamlit dashboard with SHAP-based explanations for every prediction.

## Key Results
- **Best model:** Gradient Boosting, ROC-AUC 0.8393 on a customer-level held-out test set
- **Class imbalance handled explicitly:** balanced class weights (Logistic Regression, Random Forest) and sample weighting (Gradient Boosting), since only ~2% of customers fall into the high-risk proxy class
- **Data leakage identified and corrected:** an earlier version of this pipeline leaked target-construction features (transaction count, total amount) back into the model inputs, producing an artificial 1.0 AUC; the corrected pipeline excludes these and evaluates on a genuinely unseen, customer-stratified split
- **Full prediction explainability** via SHAP for every applicant, not just aggregate feature importance

## Quick Start
```bash
git clone https://github.com/sosena2/credit-risk-model
cd credit-risk-model
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt


python -m src.data_processing
python -m src.train
pytest tests/ -v
streamlit run dashboard/app.py
```

## Project Structure
├── src/
│ ├── data_processing.py # feature engineering + proxy risk label
│ ├── train.py # model training, MLflow tracking
│ ├── predict.py
│ ├── explain.py # SHAP explainability
│ └── config.py
├── dashboard/
│ └── app.py # Streamlit risk-scoring interface
├── tests/
│ └── test_pipeline.py
├── notebooks/
│ └── eda.ipynb
├── data/
│ ├── raw/
│ └── processed/
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
└── requirements.txt


## Demo
[Link to dashboard screenshots/GIF]

## Technical Details
- **Data:** Xente eCommerce transaction data; RFM features engineered per customer
- **Proxy target:** Since real default labels don't exist, `is_high_risk` is constructed from a percentile-rank composite of recency, frequency, and monetary value — the bottom-engagement quartile is labeled high risk
- **Model:** Best of {Logistic Regression, Random Forest, Gradient Boosting}, selected by ROC-AUC among models with non-zero F1, tracked via MLflow
- **Evaluation:** ROC-AUC, precision/recall/F1 on a held-out, customer-stratified test set (no customer's transactions appear in both train and test)
- **Explainability:** SHAP waterfall plots per prediction, surfaced live in the dashboard

## Future Improvements
- Replace the proxy label with real default outcomes once available
- Tune the decision threshold explicitly for business cost trade-offs rather than the default 0.5
- Add model monitoring for drift as new transaction patterns emerge
- A/B test the credit policy threshold against real approval outcomes

## Author
Sosena — [GitHub: sosena2](https://github.com/sosena2) 
