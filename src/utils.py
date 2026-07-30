from __future__ import annotations
import pandas as pd
from typing import Tuple
from sklearn.cluster import KMeans
from .config import ModelConfig


def compute_rfm(
    df: pd.DataFrame,
    customer_id_col: str,
    date_col: str,
    amount_col: str,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary metrics per customer.

    Args:
        df: Transaction-level dataframe.
        customer_id_col: Column identifying the customer.
        date_col: Column with transaction timestamps.
        amount_col: Column with transaction amounts.
        snapshot_date: Reference date for recency calculation.

    Returns:
        DataFrame indexed by customer with recency, frequency, monetary columns.
    """
    grouped = df.groupby(customer_id_col).agg(
        recency=(date_col, lambda x: (snapshot_date - x.max()).days),
        frequency=(amount_col, "count"),
        monetary=(amount_col, "sum"),
    )
    return grouped.reset_index()


def assign_proxy_target(
    rfm_df: pd.DataFrame,
    config: ModelConfig,
) -> pd.DataFrame:
    """
    Cluster customers on RFM features and label the least-engaged,
    lowest-spend cluster as high risk (proxy for default).

    Args:
        rfm_df: DataFrame with recency, frequency, monetary columns.
        config: Pipeline configuration.

    Returns:
        rfm_df with an added `is_high_risk` binary column.
    """
    features = rfm_df[[config.rfm_recency_col, config.rfm_frequency_col, config.rfm_monetary_col]]
    normalized = (features - features.mean()) / features.std()

    kmeans = KMeans(n_clusters=config.n_clusters, random_state=config.random_state, n_init=10)
    rfm_df = rfm_df.copy()
    rfm_df["cluster"] = kmeans.fit_predict(normalized)

    cluster_means = rfm_df.groupby("cluster")[config.rfm_monetary_col].mean()
    high_risk_cluster = cluster_means.idxmin()
    rfm_df[config.target_column] = (rfm_df["cluster"] == high_risk_cluster).astype(int)

    return rfm_df.drop(columns=["cluster"])


def train_test_split_features(
    df: pd.DataFrame,
    config: ModelConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features/target into train and test sets using config settings."""
    from sklearn.model_selection import train_test_split

    feature_cols = config.numeric_features + config.categorical_features
    X = df[feature_cols]
    y = df[config.target_column]
    return train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )
