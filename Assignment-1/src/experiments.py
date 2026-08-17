"""
experiments.py
----------------
Runs every "required experiment" listed in the assignment brief as an
explicit, isolated comparison, and saves the results/plots into
results/experiment_plots/ and results/experiments_summary.json.

Run from the project root (after train.py, or standalone):
    python src/experiments.py
"""

import json
import warnings

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_prep import build_preprocessor, load_and_inspect, split_data
from utils import evaluate_split, fit_and_time, plot_bar_comparison

warnings.filterwarnings("ignore")
RANDOM_SEED = 42


def run_pipeline(model, X_train, y_train, X_val, y_val, scale_numeric=True):
    pre = build_preprocessor(scale_numeric=scale_numeric)
    pipe = Pipeline(steps=[("preprocessor", pre), ("model", model)])
    pipe, train_time = fit_and_time(pipe, X_train, y_train)
    train_metrics, _, _ = evaluate_split(pipe, X_train, y_train, "train")
    val_metrics, _, _ = evaluate_split(pipe, X_val, y_val, "validation")
    return pipe, train_time, train_metrics, val_metrics


def main():
    df = load_and_inspect("data/telecom_churn.csv", verbose=False)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    summary = {}

    # ------------------------------------------------------------------
    # 1. KNN: compare at least two values of K
    # ------------------------------------------------------------------
    print("\n[1] KNN — comparing K values")
    knn_results = []
    for k in [3, 5, 11, 25, 51]:
        pipe, t, train_m, val_m = run_pipeline(
            KNeighborsClassifier(n_neighbors=k), X_train, y_train, X_val, y_val
        )
        knn_results.append({"k": k, "train_f1": train_m["f1"], "val_f1": val_m["f1"],
                             "val_accuracy": val_m["accuracy"]})
    knn_df = pd.DataFrame(knn_results)
    knn_df.to_csv("results/experiment_plots/knn_k_comparison.csv", index=False)
    plot_bar_comparison(
        [f"k={k}" for k in knn_df["k"]], knn_df["val_f1"],
        "KNN: Validation F1 vs K", "F1-score", "results/experiment_plots/knn_k_comparison.png",
    )
    summary["knn_k_comparison"] = knn_df.to_dict(orient="records")
    print(knn_df)

    # ------------------------------------------------------------------
    # 2. Decision Tree: shallow vs deep (also serves as underfit/overfit demo)
    # ------------------------------------------------------------------
    print("\n[2] Decision Tree — shallow vs deep (underfit vs overfit)")
    dt_results = []
    for label, depth in [("shallow (depth=2)", 2), ("moderate (depth=6)", 6),
                          ("deep (depth=None)", None)]:
        pipe, t, train_m, val_m = run_pipeline(
            DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_SEED),
            X_train, y_train, X_val, y_val,
        )
        dt_results.append({
            "tree": label, "train_f1": train_m["f1"], "val_f1": val_m["f1"],
            "train_accuracy": train_m["accuracy"], "val_accuracy": val_m["accuracy"],
            "gap_accuracy": train_m["accuracy"] - val_m["accuracy"],
        })
    dt_df = pd.DataFrame(dt_results)
    dt_df.to_csv("results/experiment_plots/decision_tree_depth_comparison.csv", index=False)
    print(dt_df)
    summary["decision_tree_depth_comparison"] = dt_df.to_dict(orient="records")
    summary["underfit_overfit_demo"] = {
        "underfitting_example": "Decision Tree depth=2 — low train AND validation F1 (high bias)",
        "overfitting_example": "Decision Tree depth=None (unlimited) — high train F1, "
                                "much lower validation F1 (high variance, large train/val gap)",
        "evidence": dt_df.to_dict(orient="records"),
    }

    # ------------------------------------------------------------------
    # 3. Decision Tree vs Random Forest
    # ------------------------------------------------------------------
    print("\n[3] Decision Tree vs Random Forest")
    _, _, dt_train_m, dt_val_m = run_pipeline(
        DecisionTreeClassifier(max_depth=6, random_state=RANDOM_SEED), X_train, y_train, X_val, y_val
    )
    _, _, rf_train_m, rf_val_m = run_pipeline(
        RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1),
        X_train, y_train, X_val, y_val,
    )
    dt_vs_rf = pd.DataFrame([
        {"model": "Decision Tree", "val_f1": dt_val_m["f1"], "val_accuracy": dt_val_m["accuracy"],
         "val_roc_auc": dt_val_m["roc_auc"]},
        {"model": "Random Forest", "val_f1": rf_val_m["f1"], "val_accuracy": rf_val_m["accuracy"],
         "val_roc_auc": rf_val_m["roc_auc"]},
    ])
    dt_vs_rf.to_csv("results/experiment_plots/dt_vs_rf.csv", index=False)
    print(dt_vs_rf)
    summary["decision_tree_vs_random_forest"] = dt_vs_rf.to_dict(orient="records")

    # ------------------------------------------------------------------
    # 4. Random Forest vs Boosting (Gradient Boosting)
    # ------------------------------------------------------------------
    print("\n[4] Random Forest vs Gradient Boosting")
    _, _, gb_train_m, gb_val_m = run_pipeline(
        GradientBoostingClassifier(random_state=RANDOM_SEED), X_train, y_train, X_val, y_val
    )
    rf_vs_gb = pd.DataFrame([
        {"model": "Random Forest", "val_f1": rf_val_m["f1"], "val_roc_auc": rf_val_m["roc_auc"]},
        {"model": "Gradient Boosting", "val_f1": gb_val_m["f1"], "val_roc_auc": gb_val_m["roc_auc"]},
    ])
    rf_vs_gb.to_csv("results/experiment_plots/rf_vs_boosting.csv", index=False)
    print(rf_vs_gb)
    summary["random_forest_vs_boosting"] = rf_vs_gb.to_dict(orient="records")

    # ------------------------------------------------------------------
    # 5. SVM with vs without feature scaling
    # ------------------------------------------------------------------
    print("\n[5] SVM — with vs without feature scaling")
    _, _, svm_scaled_train, svm_scaled_val = run_pipeline(
        SVC(probability=True, kernel="rbf", random_state=RANDOM_SEED),
        X_train, y_train, X_val, y_val, scale_numeric=True,
    )
    _, _, svm_unscaled_train, svm_unscaled_val = run_pipeline(
        SVC(probability=True, kernel="rbf", random_state=RANDOM_SEED),
        X_train, y_train, X_val, y_val, scale_numeric=False,
    )
    svm_scaling = pd.DataFrame([
        {"setup": "SVM with scaling", "val_f1": svm_scaled_val["f1"],
         "val_accuracy": svm_scaled_val["accuracy"], "val_roc_auc": svm_scaled_val["roc_auc"]},
        {"setup": "SVM without scaling", "val_f1": svm_unscaled_val["f1"],
         "val_accuracy": svm_unscaled_val["accuracy"], "val_roc_auc": svm_unscaled_val["roc_auc"]},
    ])
    svm_scaling.to_csv("results/experiment_plots/svm_scaling_comparison.csv", index=False)
    plot_bar_comparison(
        svm_scaling["setup"], svm_scaling["val_f1"],
        "SVM: Effect of Feature Scaling on Validation F1", "F1-score",
        "results/experiment_plots/svm_scaling_comparison.png", color="indianred",
    )
    print(svm_scaling)
    summary["svm_scaling_comparison"] = svm_scaling.to_dict(orient="records")

    # ------------------------------------------------------------------
    # 6. Class imbalance effect: default vs class_weight='balanced'
    # ------------------------------------------------------------------
    print("\n[6] Class imbalance — default vs class_weight='balanced' (Logistic Regression & RF)")
    imbalance_results = []
    for name, model in [
        ("LogReg (default)", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ("LogReg (balanced)", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)),
        ("RandomForest (default)", RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)),
        ("RandomForest (balanced)", RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)),
    ]:
        _, _, train_m, val_m = run_pipeline(model, X_train, y_train, X_val, y_val)
        imbalance_results.append({
            "setup": name, "val_precision": val_m["precision"], "val_recall": val_m["recall"],
            "val_f1": val_m["f1"], "val_accuracy": val_m["accuracy"],
        })
    imbalance_df = pd.DataFrame(imbalance_results)
    imbalance_df.to_csv("results/experiment_plots/class_imbalance_comparison.csv", index=False)
    plot_bar_comparison(
        imbalance_df["setup"], imbalance_df["val_recall"],
        "Effect of class_weight='balanced' on Recall (churn class)", "Recall",
        "results/experiment_plots/class_imbalance_recall.png", color="mediumpurple",
    )
    print(imbalance_df)
    summary["class_imbalance_comparison"] = imbalance_df.to_dict(orient="records")

    with open("results/experiments_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nSaved results/experiments_summary.json and all experiment CSV/PNG outputs.")


if __name__ == "__main__":
    main()
