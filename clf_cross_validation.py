"""
Classification cross-validation module: orchestrates 5-fold CV for the
malignant/benign classifier in either mask-aware or independent mode.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

import config
from clf_dataset import (
    compute_class_weights,
    extract_label,
    get_clf_fold_dataloaders,
)
from clf_model import create_classifier
from clf_train import train_clf_fold


def run_classification_cv(
    all_pairs: List[Tuple[str, str]],
    fold_splits: List[Tuple[np.ndarray, np.ndarray]],
    mode: str = "independent",
) -> Dict[str, Any]:
    """
    Execute 5-fold cross-validation for the classifier.

    Parameters
    ----------
    all_pairs : list of (image_path, geojson_path)
    fold_splits : list of (train_idx, val_idx) — same splits as U-Net.
    mode : ``"mask_aware"`` or ``"independent"``.

    Returns
    -------
    results : dict with keys:
        fold_metrics      – list[dict], one per fold (best val metrics)
        mean_metrics      – dict, mean across folds
        std_error_metrics – dict, standard error (std / sqrt(K))
        best_fold_idx     – int, fold with highest AUROC
        loss_histories    – list[list[dict]], per-fold epoch logs
        variant           – str, the mode used
    """
    variant_label = "Mask-Aware" if mode == "mask_aware" else "Independent"
    print(f"\n{'═' * 60}")
    print(f"  Classification CV — {variant_label}")
    print(f"{'═' * 60}\n")

    fold_metrics: List[Dict[str, float]] = []
    loss_histories: List[List[Dict]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits, start=1):
        print(f"\n{'─' * 60}")
        print(
            f"  CLF ({variant_label}) FOLD {fold_idx}/{len(fold_splits)}  "
            f"(train={len(train_idx)}, val={len(val_idx)})"
        )
        print(f"{'─' * 60}")

        # Class weights from training split only
        train_pairs = [all_pairs[i] for i in train_idx]
        class_weights = compute_class_weights(train_pairs)

        # Class distribution info
        n_mal_train = sum(extract_label(p[0]) for p in train_pairs)
        n_ben_train = len(train_pairs) - n_mal_train
        print(f"  Train class distribution: {n_ben_train} benign, "
              f"{n_mal_train} malignant")

        train_loader, val_loader = get_clf_fold_dataloaders(
            all_pairs, train_idx, val_idx, mode=mode,
        )

        model = create_classifier(mode=mode)
        best_m, epoch_logs = train_clf_fold(
            fold_idx, model, train_loader, val_loader,
            config.DEVICE, class_weights, variant=mode,
        )

        fold_metrics.append(best_m)
        loss_histories.append(epoch_logs)

        print(
            f"  ✓ Fold {fold_idx} best — "
            f"Acc: {best_m['accuracy']:.4f}  "
            f"AUROC: {best_m['auroc']:.4f}  "
            f"AUPRC: {best_m['auprc']:.4f}"
        )

    # ── Aggregate metrics ──────────────────────
    metric_names = list(fold_metrics[0].keys())
    mean_metrics: Dict[str, float] = {}
    std_error_metrics: Dict[str, float] = {}

    for name in metric_names:
        values = np.array([fm[name] for fm in fold_metrics])
        mean_metrics[name] = float(values.mean())
        std_error_metrics[name] = float(
            values.std(ddof=1) / np.sqrt(len(values))
        )

    best_fold_idx = int(
        np.argmax([fm["auroc"] for fm in fold_metrics])
    ) + 1

    # ── Summary ────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  CLASSIFICATION CV RESULTS — {variant_label}")
    print(f"{'═' * 60}")
    summary_keys = ["accuracy", "auroc", "auprc", "f1_weighted"]
    for name in summary_keys:
        if name in mean_metrics:
            print(
                f"  {name:>15s}: {mean_metrics[name]:.4f} "
                f"± {std_error_metrics[name]:.4f}"
            )
    print(f"\n  Best fold: {best_fold_idx}")
    print(f"{'═' * 60}\n")

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_error_metrics": std_error_metrics,
        "best_fold_idx": best_fold_idx,
        "loss_histories": loss_histories,
        "variant": mode,
    }
