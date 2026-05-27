"""
train.py

Trains the High-Cost Claim Prediction model.

This version supports either:

1. A single training CSV file

   python train.py --train_csv data/high_cost_claim_training_data.csv

2. A folder containing multiple training CSV files

   python train.py --train_folder data/training_files

The multi-file option is useful when training on several years, clients,
reporting periods, or batches of healthcare claims data.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import FEATURE_COLUMNS, TARGET_COLUMN, MODEL_ARTIFACT_PATH
from src.feature_engineering import add_engineered_features
from src.data_processing import validate_training_dataset


def parse_arguments():
    """
    Defines command-line arguments for model training.
    """

    parser = argparse.ArgumentParser(
        description="Train the High-Cost Claim Prediction model."
    )

    parser.add_argument(
        "--train_csv",
        type=str,
        help="Path to a single training CSV file."
    )

    parser.add_argument(
        "--train_folder",
        type=str,
        help="Path to a folder containing multiple training CSV files."
    )

    return parser.parse_args()


def load_single_csv(csv_path):
    """
    Loads one CSV training file.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {csv_path}"
        )

    if csv_path.suffix.lower() != ".csv":
        raise ValueError(
            ".csv file needed. Please provide a valid training CSV file."
        )

    print(f"\nLoading training CSV:\n{csv_path}")

    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()

    df["source_file"] = csv_path.name

    return df


def load_training_folder(folder_path):
    """
    Loads and combines all CSV files from a folder.

    Each CSV file must follow the same required schema.
    """

    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Training folder not found: {folder_path}"
        )

    if not folder_path.is_dir():
        raise NotADirectoryError(
            f"The provided path is not a folder: {folder_path}"
        )

    csv_files = sorted(folder_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in folder: {folder_path}"
        )

    print("\nCSV files found for training:")

    dataframes = []

    for csv_file in csv_files:
        print(f"- {csv_file.name}")

        df = pd.read_csv(csv_file)

        df.columns = df.columns.str.strip()

        df["source_file"] = csv_file.name

        dataframes.append(df)

    combined_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    return combined_df


def load_training_data(args):
    """
    Determines whether to load a single CSV or a folder of CSVs.

    Exactly one option should be provided:
    - --train_csv
    - --train_folder
    """

    if args.train_csv and args.train_folder:
        raise ValueError(
            "Please provide either --train_csv OR --train_folder, not both."
        )

    if not args.train_csv and not args.train_folder:
        raise ValueError(
            "Please provide either --train_csv or --train_folder."
        )

    if args.train_csv:
        return load_single_csv(args.train_csv)

    return load_training_folder(args.train_folder)


def main():
    """
    Main training workflow.
    """

    args = parse_arguments()

    print("\n===================================")
    print(" HIGH-COST CLAIM MODEL TRAINING")
    print("===================================")

    # ============================================
    # LOAD TRAINING DATA
    # ============================================

    df = load_training_data(args)

    print("\nCombined training dataset preview:")
    print(df.head())

    print("\nTraining dataset shape:")
    print(df.shape)

    if "source_file" in df.columns:
        print("\nRows by source file:")
        print(df["source_file"].value_counts())

    # ============================================
    # VALIDATE TRAINING DATA
    # ============================================

    print("\nValidating training dataset...")

    df = validate_training_dataset(df)

    # ============================================
    # FEATURE ENGINEERING
    # ============================================

    print("\nApplying healthcare feature engineering...")

    df = add_engineered_features(df)

    # ============================================
    # FEATURE / TARGET SPLIT
    # ============================================

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # ============================================
    # TRAIN / TEST SPLIT
    # ============================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # ============================================
    # FEATURE SCALING
    # ============================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # ============================================
    # MODEL TRAINING
    # ============================================

    print("\nTraining linear regression model...")

    model = LinearRegression()

    model.fit(X_train_scaled, y_train)

    # ============================================
    # MODEL EVALUATION
    # ============================================

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

    # ============================================
    # SAVE MODEL ARTIFACT
    # ============================================

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
        },
        "training_rows": int(df.shape[0]),
        "training_columns": list(df.columns)
    }

    joblib.dump(
        artifacts,
        MODEL_ARTIFACT_PATH
    )

    print("\nModel artifact saved successfully:")
    print(MODEL_ARTIFACT_PATH)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()