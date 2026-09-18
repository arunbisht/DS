
from pathlib import Path
import sys
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

# Import is required because the saved pipeline contains FeatureEngineer.
from feature_engineering import FeatureEngineer  # noqa: F401

MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Predicts whether a telecom customer is likely to churn."
)

@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/predict")
def predict(payload: dict):
    try:
        if not isinstance(payload, dict) or not payload:
            return JSONResponse(
                status_code=400,
                content={"error": "Request body must be a non-empty JSON object."}
            )

        # Model expects a one-row DataFrame. Extra customerID is safely ignored
        # by FeatureEngineer; missing values are handled by the pipeline.
        row = pd.DataFrame([payload])

        required = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"
        ]
        missing = [c for c in required if c not in row.columns]
        if missing:
            return JSONResponse(
                status_code=422,
                content={"error": "Missing required fields", "fields": missing}
            )

        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0, 1])

        return {
            "prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability, 4)
        }

    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid input", "details": str(exc)}
        )
