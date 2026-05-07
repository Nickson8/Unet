"""
Classification metrics module: wraps TorchMetrics for binary classification
evaluation with per-class breakdown using sklearn.
"""

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


def create_clf_metrics(device: str = "cpu") -> MetricCollection:
    """
    Create a MetricCollection for binary classification.

    Metrics expect probabilities of the positive class (malignant).

    Returns
    -------
    MetricCollection with keys:
        accuracy, precision, recall, f1, auroc, auprc
    """
    metrics = MetricCollection(
        {
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "f1": BinaryF1Score(),
            "auroc": BinaryAUROC(),
            "auprc": BinaryAveragePrecision(),
        }
    )
    return metrics.to(device)


def update_clf_metrics(
    metrics: MetricCollection,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """
    Feed a batch of logits and targets into the metric accumulators.

    Parameters
    ----------
    logits : (N, 2) raw logits from the classifier
    targets : (N,) integer class labels {0, 1}
    """
    probs = torch.softmax(logits, dim=1)[:, 1]  # P(malignant)
    metrics.update(probs, targets)


def compute_and_reset_clf(metrics: MetricCollection) -> Dict[str, float]:
    """Compute accumulated metric values, reset, and return a plain dict."""
    values = metrics.compute()
    metrics.reset()
    return {k: v.item() for k, v in values.items()}


def compute_per_class_metrics(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute per-class and weighted-average precision, recall, F1.

    Parameters
    ----------
    all_preds : array of predicted class indices {0, 1}
    all_targets : array of true class indices {0, 1}

    Returns
    -------
    dict with keys:
        precision_benign, recall_benign, f1_benign,
        precision_malignant, recall_malignant, f1_malignant,
        precision_weighted, recall_weighted, f1_weighted
    """
    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=[0, 1], zero_division=0
    )
    w_precision, w_recall, w_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )
    return {
        "precision_benign": float(precisions[0]),
        "recall_benign": float(recalls[0]),
        "f1_benign": float(f1s[0]),
        "precision_malignant": float(precisions[1]),
        "recall_malignant": float(recalls[1]),
        "f1_malignant": float(f1s[1]),
        "precision_weighted": float(w_precision),
        "recall_weighted": float(w_recall),
        "f1_weighted": float(w_f1),
    }
