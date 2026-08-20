"""
data_preparation.py
--------------------
Deliverable: "Data-preparation notebook/script"

Dataset: Customer Personality Analysis (Kaggle)
https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis

Expected raw file: data/marketing_campaign.csv
The Kaggle download is semicolon-separated (sep=';'), which is why we
special-case the read below.

What this script does (mapped directly to the assignment's
"Data preparation" checklist):
  1. Inspect feature distributions and missing values
  2. Handle missing or invalid values
  3. Remove identifiers that should not affect distance calculations
  4. Engineer a small set of meaningful behavioral features (the raw
     dataset's ~28 columns aren't directly "customer behavior" metrics —
     they need to be combined first, e.g. 5 spend columns -> Total_Spend)
  5. Scale the selected numerical features
  6. Save the cleaned + scaled dataset for the analysis notebook

Output:
  data/prepared_data.csv        (cleaned, engineered, UNscaled — for cluster profiling)
  data/prepared_data_scaled.csv (the same rows, scaled — for feeding into the models)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/marketing_campaign.csv"


def load_raw(path=RAW_PATH):
    # Kaggle's file uses a tab delimiter (some versions ship ';'-delimited
    # instead — we detect which by sniffing the header row).
    with open(path, "r") as f:
        header = f.readline()
    sep = "\t" if "\t" in header else (";" if ";" in header else ",")
    df = pd.read_csv(path, sep=sep)
    return df


def inspect(df):
    print("Shape:", df.shape)
    print("\nMissing values per column (non-zero only):")
    missing = df.isna().sum()
    print(missing[missing > 0])
    print("\nDescribe (numeric):")
    print(df.describe().T[["mean", "std", "min", "max"]])


def clean_and_engineer(df):
    df = df.copy()

    # --- 1. Handle missing values -------------------------------------------------
    # Income is the only column with meaningful missingness (~1% of rows in the
    # original dataset). Since it's a small fraction, we drop those rows rather
    # than impute — imputing income (a key spending-power signal) risks quietly
    # distorting the very feature we rely on most for segmentation.
    before = len(df)
    df = df.dropna(subset=["Income"])
    print(f"Dropped {before - len(df)} rows with missing Income.")

    # --- 2. Handle invalid / outlier values ----------------------------------------
    # A few known data-entry errors in this dataset:
    #   - Year_Birth as early as 1893/1899/1900 (customers who'd be 125+ years old)
    #   - One Income value of 666,666 (a >20x outlier vs. the rest of the distribution)
    df = df[df["Year_Birth"] >= 1940]
    df = df[df["Income"] < 200000]

    # Normalize a couple of nonsense Marital_Status categories present in the raw data
    df["Marital_Status"] = df["Marital_Status"].replace(
        {"Absurd": "Other", "YOLO": "Other", "Alone": "Single"}
    )

    # --- 3. Parse dates & engineer behavioral features ------------------------------
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
    reference_date = df["Dt_Customer"].max()  # most recent enrollment date in the dataset

    df["Age"] = reference_date.year - df["Year_Birth"]
    df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days

    spend_cols = ["MntWines", "MntFruits", "MntMeatProducts",
                  "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df["Total_Spend"] = df[spend_cols].sum(axis=1)

    purchase_cols = ["NumDealsPurchases", "NumWebPurchases",
                      "NumCatalogPurchases", "NumStorePurchases"]
    df["Total_Purchases"] = df[purchase_cols].sum(axis=1)

    df["Family_Size"] = df["Kidhome"] + df["Teenhome"] + \
        df["Marital_Status"].isin(["Married", "Together"]).astype(int) + 1

    campaign_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                      "AcceptedCmp4", "AcceptedCmp5", "Response"]
    df["Total_Campaigns_Accepted"] = df[campaign_cols].sum(axis=1)

    # Guard against any negative frequency values slipping through
    df = df[df["Total_Purchases"] >= 0]

    # --- 4. Remove identifiers & non-behavioral columns -----------------------------
    # ID, raw birth year / enrollment date, and the constant Z_CostContact /
    # Z_Revenue columns (zero variance) carry no distance-relevant signal and
    # would only distort Euclidean distance calculations if left in.
    drop_cols = ["ID", "Year_Birth", "Dt_Customer", "Z_CostContact", "Z_Revenue"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df = df.reset_index(drop=True)
    return df


def select_features(df):
    """The ≥4 meaningful numeric behavioral features used for clustering."""
    features = [
        "Income",
        "Recency",
        "Total_Spend",
        "Total_Purchases",
        "Age",
        "Customer_Tenure_Days",
        "NumWebVisitsMonth",
        "Family_Size",
    ]
    return features


def scale_features(df, features):
    """
    Why scaling matters (also required as a written explanation in the assignment):
    KMeans and DBSCAN both rely on Euclidean distance. Income ranges into the
    tens of thousands while Family_Size ranges 1-5 — without scaling, Income
    alone would dominate every distance calculation and the algorithm would
    essentially cluster on income and ignore everything else. StandardScaler
    puts every feature on a comparable footing (mean 0, std 1) so each
    contributes proportionally to the notion of "similarity".
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled, columns=[f"{c}_scaled" for c in features])
    return scaled_df, scaler


def main():
    df_raw = load_raw()
    print("=== RAW DATA INSPECTION ===")
    inspect(df_raw)

    df_clean = clean_and_engineer(df_raw)
    features = select_features(df_clean)

    print("\n=== CLEANED / ENGINEERED DATA ===")
    print(df_clean[features].describe().T)

    scaled_df, scaler = scale_features(df_clean, features)

    df_clean.to_csv("data/prepared_data.csv", index=False)
    scaled_df.to_csv("data/prepared_data_scaled.csv", index=False)

    print(f"\nSaved data/prepared_data.csv        shape={df_clean.shape}")
    print(f"Saved data/prepared_data_scaled.csv  shape={scaled_df.shape}")
    print(f"\nFeatures used for clustering: {features}")


if __name__ == "__main__":
    main()
