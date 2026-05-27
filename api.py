from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import joblib

from io import BytesIO

from src.config import MODEL_ARTIFACT_PATH
from src.feature_engineering import add_engineered_features
from src.data_processing import validate_prediction_dataset


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="High-Cost Claim Prediction API",
    description=(
        "API for predicting annual paid healthcare claims "
        "from member-level healthcare utilization data."
    )
)


# ============================================================
# CORS CONFIGURATION
# ============================================================
# TEMPORARILY OPEN FOR TESTING
# Later this can be restricted to:
# - https://www.alexknowsai.com
# - https://alexknowsai.com
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "High-Cost Claim Prediction API is running.",
        "endpoint": "/predict"
    }


# ============================================================
# RISK TIER LOGIC
# ============================================================

def assign_stop_loss_risk_tier(predicted_claims):
    """
    Assigns a business-friendly risk tier
    based on predicted annual paid claims.
    """

    if predicted_claims >= 250000:

        return "Very High Risk / Potential Stop-Loss Trigger"

    elif predicted_claims >= 150000:

        return "High Risk"

    elif predicted_claims >= 75000:

        return "Moderate Risk"

    else:

        return "Lower Risk"


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    high_cost_threshold: float = 150000
):
    """
    Upload a healthcare claims CSV file
    and return predicted annual paid claims.

    Also applies:
    - risk tier classification
    - user-defined high-cost threshold
    - high-cost member flag
    """

    # ============================================
    # VALIDATE FILE TYPE
    # ============================================

    if not file.filename.endswith(".csv"):

        return {
            "error": (
                ".csv file needed. "
                "Please try again entering a file path using a .csv file."
            )
        }

    # ============================================
    # READ CSV CONTENTS
    # ============================================

    contents = await file.read()

    df = pd.read_csv(
        BytesIO(contents)
    )

    # Clean column names
    df.columns = df.columns.str.strip()

    # ============================================
    # LOAD TRAINED MODEL ARTIFACT
    # ============================================

    artifacts = joblib.load(
        MODEL_ARTIFACT_PATH
    )

    model = artifacts["model"]

    scaler = artifacts["scaler"]

    feature_columns = artifacts["feature_columns"]

    # ============================================
    # VALIDATE INPUT DATASET
    # ============================================

    df = validate_prediction_dataset(df)

    # ============================================
    # FEATURE ENGINEERING
    # ============================================

    df = add_engineered_features(df)

    # ============================================
    # BUILD FEATURE MATRIX
    # ============================================

    X = df[feature_columns]

    # ============================================
    # SCALE FEATURES
    # ============================================

    X_scaled = scaler.transform(X)

    # ============================================
    # GENERATE PREDICTIONS
    # ============================================

    predictions = model.predict(X_scaled)

    # ============================================
    # STORE PREDICTIONS
    # ============================================

    df["predicted_annual_paid_claims"] = (
        predictions.round(2)
    )

    # ============================================
    # APPLY USER THRESHOLD
    # ============================================

    df["high_cost_threshold"] = (
        high_cost_threshold
    )

    df["is_high_cost_member"] = (
        df["predicted_annual_paid_claims"] >=
        high_cost_threshold
    )

    # ============================================
    # APPLY RISK TIERS
    # ============================================

    df["risk_tier"] = df[
        "predicted_annual_paid_claims"
    ].apply(assign_stop_loss_risk_tier)

    # ============================================
    # RETURN JSON RESPONSE
    # ============================================

    return df.to_dict(
        orient="records"
    )
