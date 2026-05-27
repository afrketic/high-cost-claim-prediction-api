"""
config.py

Central configuration for the High-Cost Claim Prediction project.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_ARTIFACT_PATH = ARTIFACT_DIR / "high_cost_claim_model.pkl"

TARGET_COLUMN = "annual_paid_claims"

BASE_FEATURES = [
    "member_age",
    "chronic_condition_count",
    "prior_year_paid_claims",
    "inpatient_admissions",
    "er_visits",
    "specialty_rx_count",
    "rx_paid_claims",
    "medical_paid_claims",
    "comorbidity_score"
]

ENGINEERED_FEATURES = [
    "total_utilization_events",
    "rx_to_medical_ratio",
    "prior_claims_per_condition",
    "specialty_rx_intensity",
    "inpatient_er_interaction"
]

FEATURE_COLUMNS = BASE_FEATURES + ENGINEERED_FEATURES
