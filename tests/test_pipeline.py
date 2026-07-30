# tests/test_pipeline.py
import pandas as pd
import numpy as np
import pytest
from src.config import ModelConfig
from src.utils import compute_rfm, assign_proxy_target, train_test_split_features


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": [1, 1, 2, 2, 3],
        "transaction_date": pd.to_datetime([
            "2026-01-01", "2026-01-15", "2026-01-05", "2026-01-20", "2026-01-10"
        ]),
        "amount": [100, 200, 50, 60, 500],
    })


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig()

# --- Unit tests ---


def test_compute_rfm_shape(sample_transactions):
    """RFM output should have one row per unique customer."""
    result = compute_rfm(
        sample_transactions, "customer_id", "transaction_date", "amount",
        snapshot_date=pd.Timestamp("2026-02-01"),
    )
    assert len(result) == sample_transactions["customer_id"].nunique()


def test_compute_rfm_columns(sample_transactions):
    """RFM output must contain recency, frequency, monetary columns."""
    result = compute_rfm(
        sample_transactions, "customer_id", "transaction_date", "amount",
        snapshot_date=pd.Timestamp("2026-02-01"),
    )
    assert {"recency", "frequency", "monetary"}.issubset(result.columns)


def test_compute_rfm_frequency_correct(sample_transactions):
    """Customer 1 has 2 transactions, frequency should equal 2."""
    result = compute_rfm(
        sample_transactions, "customer_id", "transaction_date", "amount",
        snapshot_date=pd.Timestamp("2026-02-01"),
    )
    freq_customer_1 = result.loc[result["customer_id"] == 1, "frequency"].iloc[0]
    assert freq_customer_1 == 2


def test_assign_proxy_target_binary(sample_transactions, config):
    """Proxy target column must be binary (0/1)."""
    rfm = compute_rfm(
        sample_transactions, "customer_id", "transaction_date", "amount",
        snapshot_date=pd.Timestamp("2026-02-01"),
    )
    labeled = assign_proxy_target(rfm, config)
    assert set(labeled[config.target_column].unique()).issubset({0, 1})


def test_assign_proxy_target_no_nulls(sample_transactions, config):
    """Proxy target should never contain nulls."""
    rfm = compute_rfm(
        sample_transactions, "customer_id", "transaction_date", "amount",
        snapshot_date=pd.Timestamp("2026-02-01"),
    )
    labeled = assign_proxy_target(rfm, config)
    assert labeled[config.target_column].isnull().sum() == 0

# --- Integration test ---


def test_end_to_end_split_produces_expected_partitions(config):
    """Full mini-pipeline: raw df -> feature/target split with correct proportions."""
    df = pd.DataFrame({
        "recency": np.random.randint(1, 100, 50),
        "frequency": np.random.randint(1, 20, 50),
        "monetary": np.random.uniform(10, 1000, 50),
        "avg_transaction_amount": np.random.uniform(5, 500, 50),
        "product_category": np.random.choice(["A", "B"], 50),
        "channel_id": np.random.choice(["web", "app"], 50),
        "is_high_risk": np.random.choice([0, 1], 50),
    })
    X_train, X_test, y_train, y_test = train_test_split_features(df, config)
    assert len(X_train) + len(X_test) == len(df)
    assert abs(len(X_test) / len(df) - config.test_size) < 0.05
