"""
data_prep.py
-------------
Data loading, inspection, cleaning, splitting, and preprocessing-pipeline
construction for the telecom churn classification task.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42

TARGET = "churn"
ID_COLUMN = "customer_id"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "age",
    "num_support_calls",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "payment_method",
    "tech_support",
    "paperless_billing",
    "has_partner",
    "has_dependents",
    "senior_citizen",
]


def load_and_inspect(path: str, verbose: bool = True) -> pd.DataFrame:
    """Load the raw CSV and print an inspection summary required by the
    assignment (shape, feature types, target distribution, nulls, duplicates).
    Duplicate rows are dropped and the identifier column is removed, since an
    arbitrary ID should never be used as a model feature.
    """
    df = pd.read_csv(path)

    if verbose:
        print("=" * 70)
        print("DATA INSPECTION")
        print("=" * 70)
        print(f"Shape: {df.shape}")
        print(f"\nDtypes:\n{df.dtypes}")
        print(f"\nTarget distribution:\n{df[TARGET].value_counts(normalize=True)}")
        print(f"\nNull counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
        print(f"\nDuplicate rows (full row match): {df.duplicated().sum()}")

    n_dupes = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    if verbose:
        print(f"\nDropped {n_dupes} duplicate rows. Dropped identifier column '{ID_COLUMN}'.")
        print(f"Final shape after cleaning: {df.shape}")
        print("=" * 70)

    return df


def split_data(df: pd.DataFrame, seed: int = RANDOM_SEED):
    """Create train / validation / test splits (60% / 20% / 20%) using a
    fixed random seed and stratification on the target, since churn is
    moderately imbalanced. The test set is held out completely from here on
    and must never be touched during model tuning.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    # 0.25 of the remaining 80% = 20% of the original data -> val set
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=seed
    )

    print(f"Train: {X_train.shape[0]} rows | Val: {X_val.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"Train churn rate: {y_train.mean():.3f} | Val: {y_val.mean():.3f} | Test: {y_test.mean():.3f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Build the ColumnTransformer used inside every model pipeline.

    Numeric features: median imputation (robust to outliers/missingness)
      + optional StandardScaler (required for distance/margin-based models
      such as KNN and SVM; harmless but unnecessary for tree-based models).
    Categorical features: most-frequent imputation + one-hot encoding.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor
