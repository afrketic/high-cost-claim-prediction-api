"""
feature_engineering.py

Creates healthcare-relevant engineered variables for high-cost claimant modeling.
"""

import numpy as np


def add_engineered_features(df):
    """
    Adds derived utilization, pharmacy, and claims-intensity variables.

    These features are designed to mimic the kind of indicators used in
    stop-loss, payer analytics, utilization management, and high-cost claimant
    reporting.
    """

    output_df = df.copy()

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
