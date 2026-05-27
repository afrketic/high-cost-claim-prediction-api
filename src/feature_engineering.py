"""
feature_engineering.py

Creates healthcare-relevant engineered variables for high-cost claimant modeling.
"""

import numpy as np


def add_engineered_features(df):
    """
    Adds derived utilization, pharmacy, and claims-intensity variables.

    This version also supports optional prediction fields:
    - chronic_condition_count
    - comorbidity_score

    If those fields are missing from an uploaded prediction file, the system
    creates estimated proxy values using available utilization and paid-claims
    indicators.
    """

    output_df = df.copy()

    # ============================================================
    # OPTIONAL FIELD PROXY LOGIC
    # ============================================================
    # If chronic_condition_count is unavailable, estimate it using:
    # - prior-year paid claims
    # - inpatient admissions
    # - ER visits
    #
    # This keeps the model usable even when users do not have full clinical
    # condition data available.
    # ============================================================

    if "chronic_condition_count" not in output_df.columns:
        output_df["chronic_condition_count"] = (
            (output_df["prior_year_paid_claims"] / 50000)
            + output_df["inpatient_admissions"]
            + (output_df["er_visits"] * 0.5)
        ).round(0)

    # Keep estimated chronic counts within a realistic synthetic range.
    output_df["chronic_condition_count"] = output_df[
        "chronic_condition_count"
    ].clip(lower=0, upper=10)

    # ============================================================
    # If comorbidity_score is unavailable, estimate it using:
    # - chronic condition count
    # - inpatient admissions
    # - ER visits
    # - specialty Rx count
    #
    # This creates a claims/utilization-based risk proxy.
    # ============================================================

    if "comorbidity_score" not in output_df.columns:
        output_df["comorbidity_score"] = (
            output_df["chronic_condition_count"] * 0.7
            + output_df["inpatient_admissions"] * 0.6
            + output_df["er_visits"] * 0.3
            + output_df["specialty_rx_count"] * 0.4
        ).round(2)

    # Keep estimated comorbidity scores within a realistic synthetic range.
    output_df["comorbidity_score"] = output_df[
        "comorbidity_score"
    ].clip(lower=0, upper=10)

    # ============================================================
    # ENGINEERED FEATURES
    # ============================================================

    # Combined count of high-impact utilization events.
    output_df["total_utilization_events"] = (
        output_df["inpatient_admissions"] +
        output_df["er_visits"]
    )

    # Pharmacy-to-medical paid claims ratio.
    # This can help identify members whose costs are driven more heavily by pharmacy.
    output_df["rx_to_medical_ratio"] = np.where(
        output_df["medical_paid_claims"] != 0,
        output_df["rx_paid_claims"] / output_df["medical_paid_claims"],
        0
    )

    # Prior paid claims divided by chronic condition burden.
    # Add 1 to avoid division by zero for members with no chronic conditions.
    output_df["prior_claims_per_condition"] = (
        output_df["prior_year_paid_claims"] /
        (output_df["chronic_condition_count"] + 1)
    )

    # Measures specialty-drug intensity relative to condition burden.
    output_df["specialty_rx_intensity"] = (
        output_df["specialty_rx_count"] *
        output_df["comorbidity_score"]
    )

    # Interaction between inpatient events and emergency room utilization.
    output_df["inpatient_er_interaction"] = (
        output_df["inpatient_admissions"] *
        output_df["er_visits"]
    )

    return output_df