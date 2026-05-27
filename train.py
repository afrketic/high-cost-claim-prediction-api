"""
train.py

Trains the High-Cost Claim Prediction model.

Run:
    python train.py --train_csv data/high_cost_claim_training_data.csv
"""

import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import FEATURE_COLUMNS, TARGET_COLUMN, MODEL_ARTIFACT_PATH
from src.feature_engineering import add_engineered_features
from src.data_processing import validate_training_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train the High-Cost Claim Prediction model."
    )

    parser.add_argument(
        "--train_csv",
        required=True,
        type=str,
        help="Path to the training CSV file."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("\nLoading high-cost claim training dataset...")
    df = pd.read_csv(args.train_csv)
    df.columns = df.columns.str.strip()

    print("\nTraining dataset preview:")
    print(df.head())

    print("\nValidating training dataset...")
    df = validate_training_dataset(df)

    print("\nApplying healthcare feature engineering...")
    df = add_engineered_features(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining linear regression model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation:")
    print(f"MAE  : ${mae:,.2f}")
    print(f"MSE  : {mse:,.2f}")
    print(f"RMSE : ${rmse:,.2f}")
    print(f"R²   : {r2:.4f}")

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

    print("\nModel artifact saved successfully:")
    print(MODEL_ARTIFACT_PATH)


if __name__ == "__main__":
    main()
