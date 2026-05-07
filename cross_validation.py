"""
Cross-validation module: orchestrates 5-fold CV, collects per-fold
metrics, identifies the best fold, and generates segmentation examples.
"""

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold

import config
from dataset import discover_image_mask_pairs, get_fold_dataloaders
from model import create_model
from train import train_fold


def run() -> Dict[str, Any]:
    """
    Execute 5-fold cross-validation end-to-end.

    Returns
    -------
    results : dict with keys:
        fold_metrics      – list[dict], one per fold (best val metrics)
        mean_metrics      – dict, mean across folds
        std_error_metrics – dict, standard error (std / sqrt(K))
        best_fold_idx     – int, fold with highest Dice
        dataset_info      – dict with total / per-fold counts
        loss_histories    – list[list[dict]], per-fold epoch logs
        all_pairs         – list of (img_path, geojson_path)
    """
    # ── Discover data ──────────────────────────
    all_pairs = discover_image_mask_pairs(config.DATA_DIR)[:10] + discover_image_mask_pairs(config.DATA_DIR)[-10:]
    n_total = len(all_pairs)
    print(f"\n{'═' * 60}")
    print(f"  Dataset: {n_total} image–annotation pairs found")
    print(f"{'═' * 60}\n")

    kf = KFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )

    # Store fold splits for reuse by classifiers
    fold_splits = list(kf.split(all_pairs))

    fold_metrics: List[Dict[str, float]] = []
    loss_histories: List[List[Dict]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits, start=1):
        print(f"\n{'─' * 60}")
        print(f"  FOLD {fold_idx}/{config.NUM_FOLDS}  "
              f"(train={len(train_idx)}, val={len(val_idx)})")
        print(f"{'─' * 60}")

        train_loader, val_loader = get_fold_dataloaders(
            all_pairs, train_idx, val_idx
        )

        model = create_model()
        best_metrics, epoch_logs = train_fold(
            fold_idx, model, train_loader, val_loader, config.DEVICE,
        )

        fold_metrics.append(best_metrics)
        loss_histories.append(epoch_logs)

        print(f"  ✓ Fold {fold_idx} best — "
              f"Dice: {best_metrics['dice']:.4f}  "
              f"IoU: {best_metrics['iou']:.4f}")

    # ── Aggregate metrics ──────────────────────
    metric_names = list(fold_metrics[0].keys())
    mean_metrics: Dict[str, float] = {}
    std_error_metrics: Dict[str, float] = {}

    for name in metric_names:
        values = np.array([fm[name] for fm in fold_metrics])
        mean_metrics[name] = float(values.mean())
        std_error_metrics[name] = float(values.std(ddof=1) / np.sqrt(len(values)))

    best_fold_idx = int(np.argmax([fm["dice"] for fm in fold_metrics])) + 1

    # ── Summary ────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  CROSS-VALIDATION RESULTS")
    print(f"{'═' * 60}")
    for name in metric_names:
        print(f"  {name:>10s}: {mean_metrics[name]:.4f} ± {std_error_metrics[name]:.4f}")
    print(f"\n  Best fold: {best_fold_idx}")
    print(f"{'═' * 60}\n")

    # ── Dataset info ───────────────────────────
    # Compute a representative split size (from the last fold)
    n_train_per_fold = n_total - n_total // config.NUM_FOLDS
    n_val_per_fold = n_total // config.NUM_FOLDS

    results = {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_error_metrics": std_error_metrics,
        "best_fold_idx": best_fold_idx,
        "dataset_info": {
            "total": n_total,
            "train_per_fold": n_train_per_fold,
            "val_per_fold": n_val_per_fold,
        },
        "loss_histories": loss_histories,
        "all_pairs": all_pairs,
        "fold_splits": fold_splits,
    }
    return results
