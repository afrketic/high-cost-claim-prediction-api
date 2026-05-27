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


REQUIRED_PREDICTION_COLUMNS = [
    "member_age",
    "prior_year_paid_claims",
    "inpatient_admissions",
    "er_visits",
    "specialty_rx_count",
    "rx_paid_claims",
    "medical_paid_claims"
]

OPTIONAL_PREDICTION_COLUMNS = [
    "chronic_condition_count",
    "comorbidity_score"
]


@app.get("/")
def home():
    return {
        "message": "High-Cost Claim Prediction API is running.",
        "endpoint": "/predict"
    }


def create_error_workbook(error_rows):
    output = BytesIO()

    error_df = pd.DataFrame(error_rows)

    required_df = pd.DataFrame({
        "required_columns": REQUIRED_PREDICTION_COLUMNS
    })

    optional_df = pd.DataFrame({
        "optional_columns": OPTIONAL_PREDICTION_COLUMNS
    })

    instructions_df = pd.DataFrame({
        "instructions": [
            "Upload rejected.",
            "Please use the downloadable TEMPLATE_HCC_MBRS.csv file.",
            "Each uploaded CSV must include all required columns.",
            "Optional columns may be included if available.",
            "If optional columns are missing, the model will estimate proxy values."
        ]
    })

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        error_df.to_excel(writer, sheet_name="Upload Errors", index=False)
        required_df.to_excel(writer, sheet_name="Required Fields", index=False)
        optional_df.to_excel(writer, sheet_name="Optional Fields", index=False)
        instructions_df.to_excel(writer, sheet_name="Instructions", index=False)

    output.seek(0)
    return output


def stream_excel_workbook(workbook, filename, status_code=200):
    return StreamingResponse(
        workbook,
        status_code=status_code,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


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
    validation_errors = []

    for uploaded_file in files:
        if not uploaded_file.filename.endswith(".csv"):
            validation_errors.append({
                "file_name": uploaded_file.filename,
                "error_type": "Invalid file type",
                "details": "Only .csv files are accepted.",
                "recommended_action": "Download and use TEMPLATE_HCC_MBRS.csv."
            })
            continue

        try:
            contents = await uploaded_file.read()
            temp_df = pd.read_csv(BytesIO(contents))
            temp_df.columns = temp_df.columns.str.strip()
        except Exception as e:
            validation_errors.append({
                "file_name": uploaded_file.filename,
                "error_type": "File read error",
                "details": str(e),
                "recommended_action": "Confirm the file is a valid CSV template."
            })
            continue

        missing_required_columns = [
            column for column in REQUIRED_PREDICTION_COLUMNS
            if column not in temp_df.columns
        ]

        if missing_required_columns:
            validation_errors.append({
                "file_name": uploaded_file.filename,
                "error_type": "Missing required columns",
                "details": ", ".join(missing_required_columns),
                "recommended_action": "Download and use TEMPLATE_HCC_MBRS.csv."
            })
            continue

        temp_df["source_file"] = uploaded_file.filename
        dataframes.append(temp_df)

    if validation_errors:
        error_workbook = create_error_workbook(validation_errors)
        return stream_excel_workbook(
            error_workbook,
            "prediction_error.xlsx",
            status_code=400
        )

    if not dataframes:
        error_workbook = create_error_workbook([
            {
                "file_name": "No valid files",
                "error_type": "No processable data",
                "details": "No uploaded file passed validation.",
                "recommended_action": "Download and use TEMPLATE_HCC_MBRS.csv."
            }
        ])
        return stream_excel_workbook(
            error_workbook,
            "prediction_error.xlsx",
            status_code=400
        )

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

    try:
        scored_df = validate_prediction_dataset(
            combined_df
        )
    except Exception as e:
        error_workbook = create_error_workbook([
            {
                "file_name": "Combined upload",
                "error_type": "Schema validation error",
                "details": str(e),
                "recommended_action": "Download and use TEMPLATE_HCC_MBRS.csv."
            }
        ])
        return stream_excel_workbook(
            error_workbook,
            "prediction_error.xlsx",
            status_code=400
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
    ].apply(assign_stop_loss_risk_tier)

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

    return stream_excel_workbook(
        output_excel,
        "high_cost_claim_prediction_output.xlsx",
        status_code=200
    )