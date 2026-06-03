import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import PredictRequest, PredictResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Risk Scoring API")

MODEL_URI = "models:/credit_risk_best_model/latest"

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None


@app.get("/")
def root():
    return {"message": "Credit Risk Scoring API is running."}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    input_data = pd.DataFrame([request.model_dump()])
    prob = model.predict_proba(input_data)[0][1]
    label = int(prob >= 0.5)

    return PredictResponse(risk_probability=round(float(prob), 4), is_high_risk=label)