"""
utils.py
---------
Shared helpers for training a pipeline, timing it, computing the required
evaluation metrics, and producing plots.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sns.set_theme(style="whitegrid")


def fit_and_time(pipeline, X_train, y_train):
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    return pipeline, train_time


def predict_and_time(pipeline, X):
    start = time.perf_counter()
    preds = pipeline.predict(X)
    elapsed = time.perf_counter() - start
    per_sample_ms = (elapsed / len(X)) * 1000
    return preds, per_sample_ms


def get_proba(pipeline, X):
    """Return positive-class probability if the estimator supports it."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X)
        # min-max scale decision scores into a pseudo-probability for ROC-AUC
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return None


def evaluate_split(pipeline, X, y, split_name=""):
    preds, inference_ms = predict_and_time(pipeline, X)
    proba = get_proba(pipeline, X)

    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, proba) if proba is not None else np.nan,
        "inference_ms_per_sample": inference_ms,
    }
    return metrics, preds, proba


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_roc_curves(roc_data: dict, save_path):
    """roc_data: {model_name: (y_true, y_proba)}"""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, y_proba) in roc_data.items():
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_bar_comparison(labels, values, title, ylabel, save_path, color="steelblue"):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
