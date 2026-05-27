from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from typing import List
from io import BytesIO

import pandas as pd
import joblib

from src.config import MODEL_ARTIFACT_PATH
from src.feature_engineering import add_engineered_features
from src.data_processing import validate_prediction_dataset


app = FastAPI(
    title="High-Cost Claim Prediction API",
    description=(
        "Enterprise batch-processing API for predicting annual paid healthcare "
        "claims from one or more member-level healthcare CSV files."
    )
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "High-Cost Claim Prediction API is running.",
        "endpoint": "/predict"
    }


def assign_stop_loss_risk_tier(predicted_claims):
    if predicted_claims >= 250000:
        return "Very High Risk / Potential Stop-Loss Trigger"
    elif predicted_claims >= 150000:
        return "High Risk"
    elif predicted_claims >= 75000:
        return "Moderate Risk"
    else:
        return "Lower Risk"


@app.post("/predict")
async def predict(
    files: List[UploadFile] = File(...),
    high_cost_threshold: float = 150000
):
    dataframes = []

    for uploaded_file in files:

        if not uploaded_file.filename.endswith(".csv"):
            error_excel = BytesIO()

            error_df = pd.DataFrame({
                "error": [f"{uploaded_file.filename} is not a CSV file."]
            })

            with pd.ExcelWriter(error_excel, engine="openpyxl") as writer:
                error_df.to_excel(
                    writer,
                    sheet_name="Error",
                    index=False
                )

            error_excel.seek(0)

            return StreamingResponse(
                error_excel,
                media_type=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                headers={
                    "Content-Disposition":
                    "attachment; filename=prediction_error.xlsx"
                }
            )

        contents = await uploaded_file.read()

        temp_df = pd.read_csv(
            BytesIO(contents)
        )

        temp_df.columns = (
            temp_df.columns
            .str.strip()
        )

        temp_df["source_file"] = uploaded_file.filename

        dataframes.append(temp_df)

    combined_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    artifacts = joblib.load(
        MODEL_ARTIFACT_PATH
    )

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_columns = artifacts["feature_columns"]

    scored_df = validate_prediction_dataset(
        combined_df
    )

    scored_df = add_engineered_features(
        scored_df
    )

    X = scored_df[
        feature_columns
    ]

    X_scaled = scaler.transform(X)

    predictions = model.predict(
        X_scaled
    )

    scored_df["predicted_annual_paid_claims"] = (
        predictions.round(2)
    )

    scored_df["high_cost_threshold"] = high_cost_threshold

    scored_df["is_high_cost_member"] = (
        scored_df["predicted_annual_paid_claims"] >= high_cost_threshold
    )

    scored_df["risk_tier"] = scored_df[
        "predicted_annual_paid_claims"
    ].apply(
        assign_stop_loss_risk_tier
    )

    scored_df = scored_df.sort_values(
        by="predicted_annual_paid_claims",
        ascending=False
    )

    high_cost_df = scored_df[
        scored_df["is_high_cost_member"] == True
    ].copy()

    output_excel = BytesIO()

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:

        high_cost_df.to_excel(
            writer,
            sheet_name="High Cost Members",
            index=False
        )

        scored_df.to_excel(
            writer,
            sheet_name="All Scored Members",
            index=False
        )

    output_excel.seek(0)

    return StreamingResponse(
        output_excel,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        headers={
            "Content-Disposition":
            "attachment; filename=high_cost_claim_prediction_output.xlsx"
        }
    )