"""Shared loading and metrics, reused by every step.

PR-AUC (average precision) is threshold-free and is the headline metric for rare
classes. Precision, recall and F1 depend on the threshold we choose, so the
threshold is always reported next to them.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import PROCESSED, TARGET


def load_split(name: str):
    """name in {'train', 'val', 'test'}. Returns X (DataFrame), y (Series)."""
    df = pd.read_parquet(PROCESSED / f"{name}.parquet")
    return df.drop(columns=TARGET), df[TARGET]


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
