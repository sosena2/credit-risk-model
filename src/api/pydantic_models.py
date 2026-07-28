from pydantic import BaseModel


class PredictRequest(BaseModel):
    Amount: float
    Value: float
    total_transaction_amount: float
    average_transaction_amount: float
    transaction_count: int
    std_transaction_amount: float
    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_year: int


class PredictResponse(BaseModel):
    risk_probability: float
    is_high_risk: int