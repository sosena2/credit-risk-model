import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ModelConfig
from src.utils import compute_rfm, assign_proxy_target
import pandas as pd

config = ModelConfig()

df = pd.read_csv("data/data.csv")  # replace with your real path
df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], utc=True)

print("dtype check:", df["TransactionStartTime"].dtype)

snapshot_date = pd.Timestamp.now(tz="UTC")
print("snapshot_date tz check:", snapshot_date)

rfm = compute_rfm(
    df,
    customer_id_col="CustomerId",
    date_col="TransactionStartTime",
    amount_col="Amount",
    snapshot_date=snapshot_date
)

labeled = assign_proxy_target(rfm, config)
print(labeled.head())