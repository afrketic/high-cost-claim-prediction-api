"""
data_processing.py

Validation helpers for training and prediction datasets.
"""

from src.config import BASE_FEATURES, TARGET_COLUMN


def validate_training_dataset(df):
    """
    Validates that the training dataset includes all required feature columns
    plus the target variable.
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
    Validates that the prediction dataset includes the required input columns.
    The prediction dataset does not need the target column.
    """

    missing_columns = [
        column for column in BASE_FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required prediction columns: {missing_columns}"
        )

    return df.dropna(subset=BASE_FEATURES)
