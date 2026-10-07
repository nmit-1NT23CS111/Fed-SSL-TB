"""Evaluation utilities for binary TB detection with threshold-aware metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_threshold(y_true: np.ndarray, y_score: np.ndarray, method: str = "youdan") -> float:
    """Select a decision threshold using validation data only. Supports Youden J or max F1."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0:
        return 0.5

    thresholds = np.linspace(0.0, 1.0, 101)
    best_threshold = 0.5
    best_value = -np.inf

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() if confusion_matrix(y_true, y_pred, labels=[0, 1]).size == 4 else (0, 0, 0, 0)
        if method.lower() == "f1":
            value = f1_score(y_true, y_pred, zero_division=0)
        else:
            sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            value = sensitivity + specificity - 1.0
        if value > best_value:
            best_value = float(value)
            best_threshold = float(threshold)
    return float(best_threshold)


def evaluate(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float | None = None, threshold_method: str = "youdan") -> dict:
    """Compute binary classification metrics and report confusion counts and operating-point statistics."""
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float).reshape(-1)
    if y_true.size != y_pred_proba.size:
        raise ValueError(f"Mismatched true/prediction lengths: {y_true.size} vs {y_pred_proba.size}")
    if y_true.size == 0:
        return {"auc": float("nan"), "accuracy": float("nan"), "sensitivity": float("nan"), "specificity": float("nan"), "precision": float("nan"), "f1": float("nan"), "balanced_accuracy": float("nan"), "npv": float("nan"), "threshold": 0.5, "tp": 0, "tn": 0, "fp": 0, "fn": 0, "confusion_matrix": np.zeros((2, 2), dtype=int), "mean_probability": float("nan"), "min_probability": float("nan"), "max_probability": float("nan"), "positive_prediction_rate": float("nan"), "negative_prediction_rate": float("nan")}

    if threshold is None:
        threshold = select_threshold(y_true, y_pred_proba, method=threshold_method)
    y_pred = (y_pred_proba >= threshold).astype(int)

    try:
        auc = float(roc_auc_score(y_true, y_pred_proba))
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        cm = np.zeros((2, 2), dtype=int)
    tn, fp = cm[0]
    fn, tp = cm[1]

    sensitivity = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    specificity = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    balanced = float(balanced_accuracy_score(y_true, y_pred))
    npv = float(tn / (tn + fn)) if (tn + fn) else 0.0

    return {
        "auc": auc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced,
        "npv": npv,
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": cm,
        "mean_probability": float(y_pred_proba.mean()),
        "min_probability": float(y_pred_proba.min()),
        "max_probability": float(y_pred_proba.max()),
        "positive_prediction_rate": float(y_pred.mean()),
        "negative_prediction_rate": float((1 - y_pred).mean()),
    }


def format_metrics(metrics: dict) -> str:
    """Pretty-print metrics as a concise string."""
    return (
        f"AUC={metrics.get('auc', float('nan')):.4f}  "
        f"Acc={metrics.get('accuracy', float('nan')):.4f}  "
        f"Sens={metrics.get('sensitivity', float('nan')):.4f}  "
        f"Spec={metrics.get('specificity', float('nan')):.4f}  "
        f"Prec={metrics.get('precision', float('nan')):.4f}  "
        f"F1={metrics.get('f1', float('nan')):.4f}"
    )
