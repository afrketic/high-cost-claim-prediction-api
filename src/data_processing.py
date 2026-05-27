"""
data_processing.py

Validation helpers for training and prediction datasets.
"""

from src.config import BASE_FEATURES, TARGET_COLUMN


# ============================================================
# REQUIRED VS OPTIONAL PREDICTION COLUMNS
# ============================================================

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


def validate_training_dataset(df):
    """
    Validates that the training dataset includes all full model feature columns
    plus the target variable.

    Training data should remain stricter than prediction data because the model
    is learning from the complete known dataset.
    """

    required_columns = BASE_FEATURES + [TARGET_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required training columns: {missing_columns}"
        )

    return df.dropna(subset=required_columns)


def validate_prediction_dataset(df):
    """
    Validates that the prediction dataset includes the minimum required columns.

    Optional fields:
    - chronic_condition_count
    - comorbidity_score

    If optional fields are missing, they will be estimated later during
    feature engineering.
    """

    missing_required_columns = [
        column for column in REQUIRED_PREDICTION_COLUMNS
        if column not in df.columns
    ]

    if missing_required_columns:
        raise ValueError(
            f"Missing required prediction columns: {missing_required_columns}"
        )

    return df.dropna(subset=REQUIRED_PREDICTION_COLUMNS)