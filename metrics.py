"""
Metrics module: wraps TorchMetrics into a convenient collection for
binary segmentation evaluation.
"""

from typing import Dict

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)


def create_metrics(device: str = "cpu") -> MetricCollection:
    """
    Create a MetricCollection with all five required metrics.

    All metrics expect **probabilities** (after sigmoid) or raw logits —
    BinaryF1Score, BinaryJaccardIndex, etc. automatically threshold at 0.5.

    Notes
    -----
    BinaryF1Score is mathematically identical to the Dice coefficient
    for binary classification, so we use it as the Dice metric.

    Returns
    -------
    MetricCollection, keys: dice, iou, accuracy, precision, recall
    """
    metrics = MetricCollection(
        {
            "dice": BinaryF1Score(),
            "iou": BinaryJaccardIndex(),
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
        }
    )
    return metrics.to(device)


def compute_and_reset(metrics: MetricCollection) -> Dict[str, float]:
    """
    Compute accumulated metric values, reset internal state, and return
    a plain dict of {metric_name: float_value}.
    """
    values = metrics.compute()
    metrics.reset()
    return {k: v.item() for k, v in values.items()}


def update_metrics(
    metrics: MetricCollection,
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """
    Feed a batch of predictions and targets into the metric accumulators.

    Parameters
    ----------
    preds : (N, 1, H, W) raw logits
    targets : (N, 1, H, W) binary masks
    """
    # Flatten to (N*H*W,) for the Binary* metrics
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1).long()
    metrics.update(preds_flat, targets_flat)
