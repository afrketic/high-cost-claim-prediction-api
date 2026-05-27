# High-Cost Claim Prediction Streamlit Project

This project demonstrates how machine learning can support healthcare analytics, stop-loss reporting, payer intelligence, and high-cost claimant risk stratification.

## Use Case

The model predicts:

```text
annual_paid_claims
```

using member-level healthcare indicators such as:

```text
member_age
chronic_condition_count
prior_year_paid_claims
inpatient_admissions
er_visits
specialty_rx_count
rx_paid_claims
medical_paid_claims
comorbidity_score
```

## Install Packages

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train.py --train_csv data/high_cost_claim_training_data.csv
```

This creates:

```text
artifacts/high_cost_claim_model.pkl
```

## Run the Streamlit App

```bash
streamlit run app.py
```

## Test Prediction Upload

Use this file:

```text
data/high_cost_claim_prediction_input.csv
```

## App Capabilities

- Upload member-level claims data
- Predict annual paid claims
- Assign stop-loss style risk tier
- View summary metrics
- View prediction distribution
- View feature coefficient chart
- Download scored results
- Retrain model from the browser

## Important Note

This is a synthetic educational demo and should not be used for clinical, financial, underwriting, or production decision-making without proper validation, governance, privacy review, and domain expert approval.
