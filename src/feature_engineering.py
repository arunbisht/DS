
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create assignment features and perform simple input cleaning."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # ID is not a predictive business feature.
        if "customerID" in X.columns:
            X = X.drop(columns=["customerID"])

        # Convert TotalCharges from text to numeric. Invalid/blank values become NaN.
        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

        # Feature 1: average monthly spend over the customer's tenure.
        if "TotalCharges" in X.columns and "tenure" in X.columns:
            tenure = X["tenure"].replace(0, np.nan)
            X["AvgMonthlySpend"] = (X["TotalCharges"] / tenure).fillna(
                X["MonthlyCharges"] if "MonthlyCharges" in X.columns else 0
            )

        # Feature 2: number of subscribed services marked Yes.
        available = [c for c in SERVICE_COLUMNS if c in X.columns]
        if available:
            X["TotalServices"] = sum(X[c].eq("Yes").astype(int) for c in available)
        else:
            X["TotalServices"] = 0

        # Feature 3: simple tenure bands for interpretability.
        if "tenure" in X.columns:
            X["TenureGroup"] = pd.cut(
                X["tenure"],
                bins=[-1, 6, 12, 24, 48, 72],
                labels=["0-6", "7-12", "13-24", "25-48", "49-72"]
            ).astype(object)

        return X
