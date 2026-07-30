import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['is_high_risk'])
    return df


def get_features_and_target(df):
    drop_cols = [
        'CustomerId', 'is_high_risk', 'FraudResult',
        'total_transaction_amount', 'average_transaction_amount',
        'transaction_count', 'std_transaction_amount'
    ]
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['is_high_risk']
    X = X.select_dtypes(include=[np.number])
    return X, y


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }


def train():
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("credit_risk_experiment")
    logger.info("Loading data...")

    df = load_data('data/processed/processed_data.csv')
    X, y = get_features_and_target(df)

    # Stratify by customer, not by row, so no customer's transactions
    # span both train and test, and class balance is preserved exactly.
    customer_labels = df.groupby('CustomerId')['is_high_risk'].first()
    train_customers, test_customers = train_test_split(
        customer_labels.index,
        test_size=0.2,
        stratify=customer_labels.values,
        random_state=42
    )

    train_mask = df['CustomerId'].isin(train_customers)
    test_mask = df['CustomerId'].isin(test_customers)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    logger.info(f"Train class balance:\n{y_train.value_counts()}")
    logger.info(f"Test class balance:\n{y_test.value_counts()}")

    if y_test.nunique() < 2:
        logger.error("Test set has only one class present. Cannot compute ROC-AUC.")
        return

    # Sample weights for models without a native class_weight option (GradientBoosting)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    models = {
        'LogisticRegression': {
            'model': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
            ]),
            'params': {
                'clf__C': [0.01, 0.1, 1, 10],
                'clf__solver': ['lbfgs', 'liblinear']
            }
        },
        'RandomForest': {
            'model': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))
            ]),
            'params': {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [5, 10, None],
                'clf__min_samples_split': [2, 5]
            }
        },
        'GradientBoosting': {
            'model': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('clf', GradientBoostingClassifier(random_state=42))
            ]),
            'params': {
                'clf__n_estimators': [100, 200],
                'clf__learning_rate': [0.05, 0.1],
                'clf__max_depth': [3, 5]
            }
        }
    }

    best_roc_auc = 0
    best_model_name = None
    best_model = None

    for name, config in models.items():
        logger.info(f"Training {name}...")

        with mlflow.start_run(run_name=name):
            search = RandomizedSearchCV(
                config['model'],
                config['params'],
                n_iter=5,
                cv=3,
                scoring='roc_auc',
                random_state=42,
                n_jobs=-1
            )

            if name == 'GradientBoosting':
                search.fit(X_train, y_train, clf__sample_weight=sample_weights)
            else:
                search.fit(X_train, y_train)

            best_estimator = search.best_estimator_

            metrics = evaluate_model(best_estimator, X_test, y_test)

            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                best_estimator,
                artifact_path="model",
                serialization_format="pickle"
            )

            logger.info(f"{name} — ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}")

            # Require a non-zero F1 so a model that never predicts the positive
            # class can't win "best model" purely on AUC.
            if metrics['roc_auc'] > best_roc_auc and metrics['f1'] > 0:
                best_roc_auc = metrics['roc_auc']
                best_model_name = name
                best_model = best_estimator

    if best_model is None:
        logger.error("No model achieved a non-zero F1 score. Review class balance or thresholds.")
        return

    logger.info(f"Best model: {best_model_name} with ROC-AUC: {best_roc_auc:.4f}")
    with mlflow.start_run(run_name=f"best_model_{best_model_name}"):
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            registered_model_name="credit_risk_best_model",
            serialization_format="pickle"
        )
        logger.info("Best model registered in MLflow Model Registry.")


if __name__ == '__main__':
    train()
