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


def _drop_existing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [col for col in columns if col in df.columns]
    if existing:
        return df.drop(columns=existing)
    return df


def _infer_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])
    ]
    categorical_features = [col for col in df.columns if col not in numeric_features]
    return numeric_features, categorical_features


def load_and_inspect(
    path: str,
    target_column: str = "churn",
    id_columns: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load the raw CSV and print an inspection summary required by the
    assignment (shape, feature types, target distribution, nulls, duplicates).
    Duplicate rows are dropped and the identifier column is removed, since an
    arbitrary ID should never be used as a model feature.
    """
    df = pd.read_csv(path)
    id_columns = id_columns or []

    if verbose:
        print("=" * 70)
        print("DATA INSPECTION")
        print("=" * 70)
        print(f"Shape: {df.shape}")
        print(f"\nDtypes:\n{df.dtypes}")
        print(f"\nTarget distribution:\n{df[target_column].value_counts(normalize=True)}")
        print(f"\nNull counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
        print(f"\nDuplicate rows (full row match): {df.duplicated().sum()}")

    n_dupes = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    df = _drop_existing_columns(df, id_columns)

    if verbose:
        dropped_ids = ", ".join(id_columns) if id_columns else "none"
        print(f"\nDropped {n_dupes} duplicate rows. Dropped identifier columns: {dropped_ids}.")
        print(f"Final shape after cleaning: {df.shape}")
        print("=" * 70)

    return df


def split_data(df: pd.DataFrame, target_column: str = "churn", seed: int = RANDOM_SEED):
    """Create train / validation / test splits (60% / 20% / 20%) using a
    fixed random seed and stratification on the target, since churn is
    moderately imbalanced. The test set is held out completely from here on
    and must never be touched during model tuning.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

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


def build_preprocessor(
    X_reference: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build the ColumnTransformer used inside every model pipeline.

    Numeric features: median imputation (robust to outliers/missingness)
      + optional StandardScaler (required for distance/margin-based models
      such as KNN and SVM; harmless but unnecessary for tree-based models).
    Categorical features: most-frequent imputation + one-hot encoding.
    """
    numeric_features = numeric_features or []
    categorical_features = categorical_features or []

    if not numeric_features and not categorical_features:
        numeric_features, categorical_features = _infer_feature_columns(X_reference)

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
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor
