# Credit Risk Probability Model for Alternative Data

An end-to-end implementation for building, deploying, and automating a credit risk model for Bati Bank's buy-now-pay-later service.

---

## Project Structure
credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── init.py
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
│   └── test_data_processing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
---

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Capital Accord requires financial institutions to rigorously measure, justify, and document the risk models they use for credit decisions. Under Basel II, banks must demonstrate to regulators that their models are not black boxes — every input variable must be justified, every modeling assumption must be documented, and the model's outputs must be explainable to both internal risk teams and external auditors.

This regulatory expectation directly shapes how a credit scoring model must be built. A model that achieves high predictive accuracy but cannot explain why a customer was labeled high-risk is non-compliant. For example, if a gradient boosting model denies credit to an applicant, the bank must be able to articulate which factors drove that decision and why those factors are legitimate risk indicators — not proxies for protected characteristics like race or gender.

In practice, this means the model must:
- Use variables with documented business justification
- Be validated on out-of-sample data with tracked performance metrics
- Be monitored over time for model drift
- Produce outputs that a credit officer can interpret and override if needed

This is why interpretable models like Logistic Regression with Weight of Evidence (WoE) transformations remain the industry standard in regulated credit scoring, even when more complex models offer better raw performance. The documentation and interpretability requirements of Basel II are not optional — they are a legal and operational constraint.

---

### 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The raw transaction dataset from the eCommerce platform contains no explicit label indicating whether a customer has defaulted on a loan. This is common when working with alternative data sources — the data was not originally collected for credit risk purposes, so there is no loan repayment history, no missed payment record, and no formal default event to learn from.

Without a target variable, supervised machine learning cannot be applied directly. A proxy variable is therefore constructed to approximate the concept of credit risk using observable behavioral signals. In this project, Recency, Frequency, and Monetary (RFM) metrics are used to segment customers. Customers who are disengaged — meaning they transact rarely, have not transacted recently, and spend little — are labeled as high-risk (is_high_risk = 1), on the assumption that low engagement correlates with financial instability or inability to repay.

However, using a proxy variable instead of a true default label introduces several business risks:

- **Label noise:** The proxy may incorrectly label customers. A customer who is low-frequency may simply be a selective buyer, not a credit risk. This means the model learns from imperfect signal, which degrades its real-world reliability.
- **Concept mismatch:** Behavioral disengagement on an eCommerce platform is not the same as loan default. The assumption that one predicts the other is a modeling choice, not an empirical fact, and it may not hold across different customer segments or economic conditions.
- **Regulatory scrutiny:** Under Basel II, institutions must justify their target variable. A proxy variable must be clearly documented as an approximation, and the bank must acknowledge its limitations in any model risk documentation submitted to regulators.
- **Feedback loops:** If the model trained on the proxy denies credit to customers who would have actually repaid, the bank loses revenue and the model never gets corrected because no repayment data is generated for those customers.

These risks do not make the proxy approach invalid — it is a reasonable and industry-accepted method when true labels are unavailable. But they must be explicitly acknowledged, and the model should be updated as real loan repayment data becomes available over time.

---

### 3. What are the key trade-offs between a simple, interpretable model (e.g., Logistic Regression with WoE) and a high-performance model (e.g., Gradient Boosting) in a regulated financial context?

In credit risk modeling, the choice between an interpretable model and a high-performance model is not purely a technical decision — it is a business, legal, and ethical one.

**Logistic Regression with Weight of Evidence (WoE)** is the traditional approach in credit scoring and remains widely used in regulated institutions for the following reasons:
- Every coefficient in the model has a direct, interpretable meaning. A loan officer can understand why a score changed.
- WoE transformations handle non-linear relationships and missing values gracefully while maintaining a linear model structure.
- The model produces a scorecard that is auditable, stable, and easy to validate against regulatory expectations.
- It is straightforward to monitor for drift and to explain to regulators, internal risk committees, and in adverse action notices sent to applicants who are denied credit.

The main limitation is that Logistic Regression with WoE may underperform on complex, non-linear patterns in the data, which can mean leaving predictive accuracy on the table.

**Gradient Boosting models** (XGBoost, LightGBM) offer the following advantages:
- They capture complex, non-linear interactions between features automatically.
- They typically outperform linear models on raw predictive metrics such as ROC-AUC and F1 score.
- They handle imbalanced datasets and noisy features more robustly.

However, in a regulated financial context, these advantages come with significant trade-offs:
- The model is not inherently interpretable. Feature importance scores provide some insight, but they do not explain individual predictions in the way a scorecard does.
- Tools like SHAP values can approximate explanations, but regulators may not accept post-hoc explanation tools as a substitute for inherent interpretability.
- Gradient boosting models are harder to validate, harder to monitor, and harder to justify in model risk documentation submitted under Basel II.
- If the model is challenged legally — for example, on grounds of discriminatory lending — the bank must be able to fully explain every decision, which is difficult with an ensemble of hundreds of trees.

**In summary:** In a regulated financial context, interpretability is not just a nice-to-have — it is often a regulatory requirement. The preferred approach is to use an interpretable model as the primary production model and use a high-performance model as a benchmark or challenger model. If a gradient boosting model significantly outperforms the logistic regression, that gap must be weighed against the compliance cost and risk of deploying a less explainable system. For this project, both model types will be trained and compared, with the final deployment decision justified against these trade-offs.

---

## Setup Instructions

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-model.git
cd credit-risk-model
pip install -r requirements.txt
```

---

## Data

Download the dataset from [Kaggle — Xente Challenge](https://www.kaggle.com/competitions/xente-fraud-detection-competition/data) and place it in `data/raw/`.

Data is excluded from version control via `.gitignore`.

---

