"""
app.py

Streamlit web app for High-Cost Claim Prediction.

Purpose:
    Demonstrates how math, analytics, and AI can be applied to healthcare
    stop-loss and high-cost claimant reporting.

App Capabilities:
    1. Upload claim-level/member-level prediction CSV
    2. Generate predicted annual paid claims
    3. Assign stop-loss risk tier
    4. Visualize predicted cost distribution
    5. Download scored output
    6. Upload training CSV
    7. Retrain the model from the browser
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    MODEL_ARTIFACT_PATH,
    FEATURE_COLUMNS,
    BASE_FEATURES,
    TARGET_COLUMN
)
from src.feature_engineering import add_engineered_features
from src.data_processing import (
    validate_prediction_dataset,
    validate_training_dataset
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def assign_stop_loss_risk_tier(predicted_claims):
    """
    Converts predicted annual paid claims into simple stop-loss style risk tiers.

    These thresholds are for demo/portfolio purposes and can be adjusted.
    """

    if predicted_claims >= 250000:
        return "Very High Risk / Potential Stop-Loss Trigger"
    elif predicted_claims >= 150000:
        return "High Risk"
    elif predicted_claims >= 75000:
        return "Moderate Risk"
    else:
        return "Lower Risk"


def plot_prediction_distribution(df):
    """
    Histogram of predicted annual paid claims.
    """

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        df["predicted_annual_paid_claims"],
        bins=10
    )

    ax.set_title("Predicted Annual Paid Claims Distribution")
    ax.set_xlabel("Predicted Annual Paid Claims")
    ax.set_ylabel("Member Count")

    return fig


def plot_feature_coefficients(model, feature_columns):
    """
    Bar chart of linear regression coefficients.
    """

    coefficient_df = pd.DataFrame({
        "Feature": feature_columns,
        "Coefficient": model.coef_
    })

    coefficient_df = coefficient_df.sort_values(
        "Coefficient",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        coefficient_df["Feature"],
        coefficient_df["Coefficient"]
    )

    ax.set_title("Linear Regression Feature Coefficients")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Coefficient Value")
    ax.tick_params(axis="x", rotation=75)

    return fig


def plot_actual_vs_predicted(y_test, predictions):
    """
    Actual vs predicted plot used after retraining.
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(y_test, predictions)

    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )

    ax.set_title("Actual vs Predicted Annual Paid Claims")
    ax.set_xlabel("Actual Annual Paid Claims")
    ax.set_ylabel("Predicted Annual Paid Claims")

    return fig


def plot_residuals(y_test, predictions):
    """
    Residual plot used after retraining.
    """

    residuals = y_test - predictions

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(predictions, residuals)

    ax.axhline(
        y=0,
        linestyle="--"
    )

    ax.set_title("Residual Plot")
    ax.set_xlabel("Predicted Annual Paid Claims")
    ax.set_ylabel("Residual")

    return fig


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="High-Cost Claim Prediction",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# APP HEADER
# ============================================================

st.title("🏥 High-Cost Claim Prediction App")

st.write(
    """
    This healthcare AI demo predicts future annual paid claims using member-level
    utilization, pharmacy, medical, and comorbidity indicators.

    It is designed to mirror the type of analytics used in payer reporting,
    stop-loss monitoring, high-cost claimant review, and risk stratification.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Required Prediction Columns")

for column in BASE_FEATURES:
    st.sidebar.write(f"- `{column}`")

st.sidebar.info(
    """
    First train the model using:

    `python train.py --train_csv data/high_cost_claim_training_data.csv`
    """
)


# ============================================================
# MODEL STATUS
# ============================================================

st.subheader("Model Status")

if MODEL_ARTIFACT_PATH.exists():
    st.success(f"Model artifact found: {MODEL_ARTIFACT_PATH}")
else:
    st.warning("No model artifact found. Train the model first or use the retraining section below.")


# ============================================================
# PREDICTION WORKFLOW
# ============================================================

st.header("📤 Upload Member Claims CSV for Prediction")

prediction_file = st.file_uploader(
    "Upload prediction CSV",
    type=["csv"],
    key="prediction_file"
)

if prediction_file is not None:

    try:
        prediction_input_df = pd.read_csv(prediction_file)
        prediction_input_df.columns = prediction_input_df.columns.str.strip()

        st.success("Prediction CSV uploaded successfully.")

        st.subheader("Uploaded Prediction Dataset")
        st.dataframe(prediction_input_df, use_container_width=True)

        if st.button("Run High-Cost Claim Prediction"):

            artifacts = joblib.load(MODEL_ARTIFACT_PATH)

            model = artifacts["model"]
            scaler = artifacts["scaler"]
            feature_columns = artifacts["feature_columns"]

            prediction_df = validate_prediction_dataset(prediction_input_df)
            prediction_df = add_engineered_features(prediction_df)

            X_prediction = prediction_df[feature_columns]
            X_prediction_scaled = scaler.transform(X_prediction)

            predictions = model.predict(X_prediction_scaled)

            prediction_df["predicted_annual_paid_claims"] = predictions.round(2)

            prediction_df["risk_tier"] = prediction_df[
                "predicted_annual_paid_claims"
            ].apply(assign_stop_loss_risk_tier)

            st.success("Predictions generated successfully.")

            st.subheader("Prediction Results")
            st.dataframe(prediction_df, use_container_width=True)

            st.subheader("High-Cost Claim Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Average Predicted Claims",
                    f"${prediction_df['predicted_annual_paid_claims'].mean():,.2f}"
                )

            with col2:
                st.metric(
                    "Max Predicted Claims",
                    f"${prediction_df['predicted_annual_paid_claims'].max():,.2f}"
                )

            with col3:
                high_risk_count = prediction_df[
                    prediction_df["predicted_annual_paid_claims"] >= 150000
                ].shape[0]

                st.metric(
                    "High-Risk Members",
                    high_risk_count
                )

            st.subheader("Visual Analytics")

            distribution_fig = plot_prediction_distribution(prediction_df)
            st.pyplot(distribution_fig)

            coefficient_fig = plot_feature_coefficients(
                model,
                feature_columns
            )
            st.pyplot(coefficient_fig)

            csv_output = prediction_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Scored High-Cost Claim Output",
                data=csv_output,
                file_name="high_cost_claim_prediction_output.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error("Prediction failed.")
        st.error(e)

else:
    st.info("Upload a member claims CSV file to begin prediction.")


# ============================================================
# RETRAINING WORKFLOW
# ============================================================

st.divider()

st.header("🔁 Retrain High-Cost Claim Model")

st.write(
    """
    Upload a new healthcare claims training file to retrain the model.

    Required target column:
    - `annual_paid_claims`
    """
)

training_file = st.file_uploader(
    "Upload training CSV",
    type=["csv"],
    key="training_file"
)

if training_file is not None:

    try:
        training_df = pd.read_csv(training_file)
        training_df.columns = training_df.columns.str.strip()

        st.success("Training CSV uploaded successfully.")

        st.subheader("Training Dataset Preview")
        st.dataframe(training_df, use_container_width=True)

        if st.button("Retrain High-Cost Claim Model"):

            training_df = validate_training_dataset(training_df)
            training_df = add_engineered_features(training_df)

            X = training_df[FEATURE_COLUMNS]
            y = training_df[TARGET_COLUMN]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42
            )

            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = LinearRegression()
            model.fit(X_train_scaled, y_train)

            retrain_predictions = model.predict(X_test_scaled)

            mae = mean_absolute_error(y_test, retrain_predictions)
            mse = mean_squared_error(y_test, retrain_predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, retrain_predictions)

            MODEL_ARTIFACT_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            artifacts = {
                "model": model,
                "scaler": scaler,
                "feature_columns": FEATURE_COLUMNS,
                "target_column": TARGET_COLUMN,
                "metrics": {
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "R2": r2
                }
            }

            joblib.dump(artifacts, MODEL_ARTIFACT_PATH)

            st.success("Model retrained and saved successfully.")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("MAE", f"${mae:,.2f}")

            with col2:
                st.metric("RMSE", f"${rmse:,.2f}")

            with col3:
                st.metric("R²", round(r2, 4))

            with col4:
                st.metric("Training Rows", training_df.shape[0])

            st.subheader("Retraining Diagnostics")

            actual_vs_predicted_fig = plot_actual_vs_predicted(
                y_test,
                retrain_predictions
            )
            st.pyplot(actual_vs_predicted_fig)

            residual_fig = plot_residuals(
                y_test,
                retrain_predictions
            )
            st.pyplot(residual_fig)

            coefficient_fig = plot_feature_coefficients(
                model,
                FEATURE_COLUMNS
            )
            st.pyplot(coefficient_fig)

            st.info(
                "The updated model artifact is now active for future predictions."
            )

    except Exception as e:
        st.error("Retraining failed.")
        st.error(e)

else:
    st.info("Upload a training CSV if you want to retrain the model.")
