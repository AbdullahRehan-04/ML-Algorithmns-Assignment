"""
generate_data.py
-----------------
Creates a synthetic telecom customer-churn dataset with a realistic mixture
of numerical and categorical features, some missing values, some duplicate
rows, and a moderately imbalanced binary target.

The dataset is fully synthetic (no real/confidential data), but the feature
relationships are constructed so that churn is genuinely predictable from
the features (with noise), which makes the downstream benchmarking
experiments meaningful.

Run:
    python src/generate_data.py
Produces:
    data/telecom_churn.csv
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_ROWS = 6000


def generate_dataset(n_rows: int = N_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- Numerical features -------------------------------------------------
    tenure_months = rng.integers(0, 73, size=n_rows)
    monthly_charges = np.round(rng.normal(65, 25, size=n_rows).clip(15, 150), 2)
    age = rng.integers(18, 80, size=n_rows)
    num_support_calls = rng.poisson(1.3, size=n_rows)
    total_charges = np.round(monthly_charges * tenure_months * rng.normal(1.0, 0.05, size=n_rows), 2)
    total_charges = total_charges.clip(0, None)

    # ---- Categorical features -------------------------------------------------
    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n_rows, p=[0.55, 0.25, 0.20]
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n_rows, p=[0.35, 0.45, 0.20]
    )
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_rows,
        p=[0.35, 0.20, 0.225, 0.225],
    )
    tech_support = rng.choice(["Yes", "No", "No internet service"], size=n_rows, p=[0.35, 0.45, 0.20])
    paperless_billing = rng.choice(["Yes", "No"], size=n_rows, p=[0.6, 0.4])
    has_partner = rng.choice(["Yes", "No"], size=n_rows, p=[0.5, 0.5])
    has_dependents = rng.choice(["Yes", "No"], size=n_rows, p=[0.3, 0.7])
    senior_citizen = rng.choice([0, 1], size=n_rows, p=[0.84, 0.16])

    # ---- Construct churn target with real signal + noise ----------------------
    logit = (
        -2.0
        + 1.4 * (contract_type == "Month-to-month")
        - 1.1 * (contract_type == "Two year")
        + 0.9 * (internet_service == "Fiber optic")
        + 0.35 * (payment_method == "Electronic check")
        + 0.05 * num_support_calls
        - 0.03 * (tenure_months / 12)
        + 0.015 * (monthly_charges - 65) / 10
        - 0.4 * (tech_support == "Yes")
        + 0.3 * senior_citizen
        + rng.normal(0, 0.9, size=n_rows)  # noise
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(size=n_rows) < prob_churn).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(n_rows)],
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "age": age,
            "num_support_calls": num_support_calls,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "tech_support": tech_support,
            "paperless_billing": paperless_billing,
            "has_partner": has_partner,
            "has_dependents": has_dependents,
            "senior_citizen": senior_citizen,
            "churn": churn,
        }
    )

    # ---- Inject realistic messiness --------------------------------------------
    # Missing values in a few columns (simulates real-world data collection gaps)
    for col, frac in [("total_charges", 0.02), ("age", 0.015), ("tech_support", 0.01)]:
        idx = rng.choice(df.index, size=int(frac * n_rows), replace=False)
        df.loc[idx, col] = np.nan

    # A handful of duplicate rows (to be caught during data inspection)
    dup_idx = rng.choice(df.index, size=25, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/telecom_churn.csv", index=False)
    print(f"Saved {len(df)} rows to data/telecom_churn.csv")
    print(f"Churn rate: {df['churn'].mean():.3f}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Missing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
