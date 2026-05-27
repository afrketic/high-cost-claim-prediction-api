from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from io import BytesIO

from src.config import MODEL_ARTIFACT_PATH
from src.feature_engineering import add_engineered_features
from src.data_processing import validate_prediction_dataset
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="High-Cost Claim Prediction API",
    description="API for predicting annual paid claims from member-level healthcare data."
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.alexknowsai.com",
        "https://alexknowsai.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "High-Cost Claim Prediction API is running.",
        "endpoint": "/predict"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return {
            "error": ".csv file required"
        }

    contents = await file.read()

    df = pd.read_csv(BytesIO(contents))
    df.columns = df.columns.str.strip()

    artifacts = joblib.load(MODEL_ARTIFACT_PATH)

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_columns = artifacts["feature_columns"]

    df = validate_prediction_dataset(df)
    df = add_engineered_features(df)

    X = df[feature_columns]
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)

    df["predicted_annual_paid_claims"] = predictions.round(2)

    return df.to_dict(orient="records")