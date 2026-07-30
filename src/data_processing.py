import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler as RFMScaler
import datetime
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


# ── 1. Aggregate Features ───────────────────────────────────────────
class AggregateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        agg = df.groupby('CustomerId')['Amount'].agg(
            total_transaction_amount='sum',
            average_transaction_amount='mean',
            transaction_count='count',
            std_transaction_amount='std'
        ).reset_index()
        agg['std_transaction_amount'] = agg['std_transaction_amount'].fillna(0)
        df = df.merge(agg, on='CustomerId', how='left')
        return df


# ── 2. Temporal Features ────────────────────────────────────────────
class TemporalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['transaction_hour'] = df['TransactionStartTime'].dt.hour
        df['transaction_day'] = df['TransactionStartTime'].dt.day
        df['transaction_month'] = df['TransactionStartTime'].dt.month
        df['transaction_year'] = df['TransactionStartTime'].dt.year
        return df


# ── 3. Drop unused columns ──────────────────────────────────────────
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')


# ── 4. Full Pipeline ────────────────────────────────────────────────
def build_pipeline():
    categorical_features = ['ProductCategory', 'ChannelId', 'ProviderId', 'PricingStrategy']
    numerical_features = [
        'Amount', 'Value',
        'total_transaction_amount', 'average_transaction_amount',
        'transaction_count', 'std_transaction_amount',
        'transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year'
    ]

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor


def run_processing(input_path, output_path):
    df = pd.read_csv(input_path)

    # Step 1: Add aggregate features
    agg = AggregateFeatures()
    df = agg.transform(df)

    # Step 2: Add temporal features
    temp = TemporalFeatures()
    df = temp.transform(df)

    # Step 3: Drop columns not needed for modeling
    drop_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId',
                 'CurrencyCode', 'CountryCode', 'TransactionStartTime', 'ProductId']
    dropper = DropColumns(drop_cols)
    df = dropper.transform(df)

    # Step 4: Save processed data (before sklearn transform, keeping CustomerId for Task 4)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    return df


if __name__ == '__main__':
    run_processing('data/raw/data.csv', 'data/processed/processed_data.csv')

def build_rfm_and_label(input_path, output_path):
    df = pd.read_csv(input_path)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    snapshot_date = df['TransactionStartTime'].max() + datetime.timedelta(days=1)

    rfm = df.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Amount', 'sum')
    ).reset_index()

    # Composite risk score: high recency (inactive) + low frequency + low monetary = risky.
    # Rank-based (percentile) scoring is robust to outliers, unlike raw KMeans distances.
    rfm['recency_rank'] = rfm['Recency'].rank(pct=True)
    rfm['frequency_rank'] = rfm['Frequency'].rank(pct=True, ascending=False)
    rfm['monetary_rank'] = rfm['Monetary'].rank(pct=True, ascending=False)

    rfm['risk_score'] = (
        rfm['recency_rank'] + rfm['frequency_rank'] + rfm['monetary_rank']
    ) / 3

    # Bottom-engagement quartile (highest risk_score) = high risk.
    # This guarantees ~25% of customers are labeled high risk, regardless of outliers.
    threshold = rfm['risk_score'].quantile(0.75)
    rfm['is_high_risk'] = (rfm['risk_score'] >= threshold).astype(int)

    print("High-risk customer count:", rfm['is_high_risk'].sum(), "out of", len(rfm))

    processed = pd.read_csv(output_path)
    processed = processed.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
    processed.to_csv(output_path, index=False)
    print(f"is_high_risk column added. Saved to {output_path}")
    return processed


if __name__ == '__main__':
    run_processing('data/raw/data.csv', 'data/processed/processed_data.csv')
    build_rfm_and_label('data/raw/data.csv', 'data/processed/processed_data.csv')