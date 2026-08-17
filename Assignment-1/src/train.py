"""
train.py
---------
Trains all seven required models on the telecom churn dataset, evaluates
them on train / validation / test splits, records timing, and saves a
results table plus confusion matrices and ROC curves.

Run from the project root:
    python src/train.py
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import load_project_config
from data_prep import build_preprocessor, load_and_inspect, split_data
from utils import evaluate_split, fit_and_time, plot_confusion_matrix, plot_roc_curves, plot_bar_comparison

warnings.filterwarnings("ignore")

RANDOM_SEED = 42


def get_models():
    """The seven required models, each with sensible default hyperparameters.
    (Assignment 3 performs proper hyperparameter tuning on the chosen model —
    here the goal is a fair baseline comparison.)
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
        "XGBoost": XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=-1, verbosity=0
        ),
        "SVM": SVC(probability=True, kernel="rbf", random_state=RANDOM_SEED),
    }


def main():
    cfg = load_project_config()
    df = load_and_inspect(
        cfg["dataset_path"],
        target_column=cfg["target_column"],
        id_columns=cfg["id_columns"],
    )
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df,
        target_column=cfg["target_column"],
        seed=cfg["random_seed"],
    )

    preprocessor = build_preprocessor(
        X_reference=X_train,
        numeric_features=cfg["numeric_features"],
        categorical_features=cfg["categorical_features"],
        scale_numeric=True,
    )
    models = get_models()

    all_results = []
    roc_data_test = {}
    fitted_pipelines = {}

    for name, model in models.items():
        print(f"\nTraining: {name}")
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe, train_time = fit_and_time(pipe, X_train, y_train)
        fitted_pipelines[name] = pipe

        for split_name, X_split, y_split in [
            ("train", X_train, y_train),
            ("validation", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            metrics, preds, proba = evaluate_split(pipe, X_split, y_split, split_name)
            metrics["model"] = name
            metrics["train_time_s"] = train_time
            all_results.append(metrics)

            if split_name == "test":
                roc_data_test[name] = (y_split, proba)
                plot_confusion_matrix(
                    y_split, preds, name,
                    f"results/confusion_matrices/{name.replace(' ', '_').replace('(', '').replace(')', '')}.png",
                )

        print(f"  train_time={train_time:.3f}s")

    results_df = pd.DataFrame(all_results)
    col_order = ["model", "split", "accuracy", "precision", "recall", "f1", "roc_auc",
                 "train_time_s", "inference_ms_per_sample"]
    results_df = results_df[col_order]
    results_df.to_csv("results/results_table.csv", index=False)
    print("\nSaved results/results_table.csv")

    # ROC curves (test set, all models)
    plot_roc_curves(roc_data_test, "results/roc_curves_test.png")

    # Timing comparison (train time + inference time), test-split rows only
    test_rows = results_df[results_df["split"] == "test"].sort_values("model")
    plot_bar_comparison(
        test_rows["model"], test_rows["train_time_s"],
        "Training Time by Model", "Seconds", "results/training_time_comparison.png", color="darkorange",
    )
    plot_bar_comparison(
        test_rows["model"], test_rows["inference_ms_per_sample"],
        "Inference Time per Sample by Model", "Milliseconds / sample",
        "results/inference_time_comparison.png", color="seagreen",
    )
    plot_bar_comparison(
        test_rows["model"], test_rows["f1"],
        "Test F1-score by Model", "F1-score", "results/test_f1_comparison.png", color="steelblue",
    )

    # Pivot summary: test accuracy/f1/roc_auc side by side for quick reading
    summary = results_df[results_df["split"] == "test"][
        ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "train_time_s", "inference_ms_per_sample"]
    ].sort_values("f1", ascending=False)
    summary.to_csv("results/test_summary_ranked.csv", index=False)
    print("\n" + "=" * 70)
    print("TEST SET SUMMARY (ranked by F1)")
    print("=" * 70)
    print(summary.to_string(index=False))

    return results_df, fitted_pipelines, (X_train, X_val, X_test, y_train, y_val, y_test)


if __name__ == "__main__":
    main()
