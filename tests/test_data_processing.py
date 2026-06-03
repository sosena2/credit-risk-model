import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import AggregateFeatures, TemporalFeatures


def make_sample_df():
    return pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'TransactionId': ['T1', 'T2', 'T3'],
        'Amount': [100, 200, 150],
        'TransactionStartTime': ['2023-01-01 10:00:00', '2023-02-15 14:30:00', '2023-03-10 08:00:00']
    })


def test_aggregate_features_columns():
    df = make_sample_df()
    transformer = AggregateFeatures()
    result = transformer.transform(df)
    expected_cols = ['total_transaction_amount', 'average_transaction_amount',
                     'transaction_count', 'std_transaction_amount']
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_aggregate_features_values():
    df = make_sample_df()
    transformer = AggregateFeatures()
    result = transformer.transform(df)
    c1_rows = result[result['CustomerId'] == 'C1']
    assert c1_rows['total_transaction_amount'].iloc[0] == 300
    assert c1_rows['transaction_count'].iloc[0] == 2


def test_temporal_features_columns():
    df = make_sample_df()
    transformer = TemporalFeatures()
    result = transformer.transform(df)
    for col in ['transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year']:
        assert col in result.columns, f"Missing column: {col}"


def test_temporal_features_values():
    df = make_sample_df()
    transformer = TemporalFeatures()
    result = transformer.transform(df)
    assert result['transaction_hour'].iloc[0] == 10
    assert result['transaction_month'].iloc[0] == 1
    assert result['transaction_year'].iloc[0] == 2023