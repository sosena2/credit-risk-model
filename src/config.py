from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ModelConfig:
    """Central configuration for the credit risk pipeline."""
    random_state: int = 42
    test_size: float = 0.2
    target_column: str = "is_high_risk"
    rfm_recency_col: str = "recency"
    rfm_frequency_col: str = "frequency"
    rfm_monetary_col: str = "monetary"
    n_clusters: int = 3  # for RFM-based proxy target clustering
    numeric_features: List[str] = field(default_factory=lambda: [
        "recency", "frequency", "monetary", "avg_transaction_amount"
    ])
    categorical_features: List[str] = field(default_factory=lambda: [
        "product_category", "channel_id"
    ])
    model_registry_name: str = "bati-bank-credit-risk"


@dataclass(frozen=True)
class APIConfig:
    """Configuration for the FastAPI serving layer."""
    host: str = "0.0.0.0"
    port: int = 8000
    model_stage: str = "Production"


# Named constants replacing magic numbers previously scattered in the pipeline
DEFAULT_THRESHOLD: float = 0.5
HIGH_RISK_CLUSTER_LABEL: int = 1
MIN_TRANSACTIONS_PER_CUSTOMER: int = 1
