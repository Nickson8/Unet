"""
main.py — Entry point for the U-Net dental lesion segmentation pipeline.

Usage:
    python main.py                           # uses config.DATA_DIR
    UNET_DATA_DIR=/path/to/data python main.py  # override dataset path
"""

import os
import random

import numpy as np
import torch
from sklearn.model_selection import KFold

import config
import cross_validation
import report
from dataset import discover_image_mask_pairs


def seed_everything(seed: int) -> None:
    """Set seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    # ── 1. Reproducibility ─────────────────────
    seed_everything(config.SEED)

    # ── 2. Create output directories ───────────
    for d in [
        config.OUTPUT_DIR,
        config.AUGMENTATION_EXAMPLES_DIR,
        config.BEST_FOLD_PREDICTIONS_DIR,
        config.CHECKPOINTS_DIR,
        config.LOGS_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    # ── 3. Discover dataset ────────────────────
    all_pairs = discover_image_mask_pairs(config.DATA_DIR)
    print(f"\n  Dataset path: {config.DATA_DIR}")
    print(f"  Total images: {len(all_pairs)}")
    print(f"  Device: {config.DEVICE}\n")

    # ── 4. Save augmentation examples ──────────
    print("Generating augmentation examples...")
    aug_paths = report.save_augmentation_examples(all_pairs, n_examples=5)

    # ── 5. Run 5-fold cross-validation ─────────
    print("\nStarting 5-fold cross-validation...\n")
    results = cross_validation.run()

    # ── 6. Generate prediction examples ────────
    #    Re-create the fold splits to get the validation indices
    #    of the best fold.
    print("\nGenerating prediction examples from best fold...")
    kf = KFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )
    best_val_indices = None
    for fold_idx, (_, val_idx) in enumerate(kf.split(results["all_pairs"]), start=1):
        if fold_idx == results["best_fold_idx"]:
            best_val_indices = val_idx
            break

    pred_paths = report.save_prediction_examples(
        results["all_pairs"],
        best_val_indices,
        results["best_fold_idx"],
        n_examples=5,
    )

    # ── 7. Plot loss curves ────────────────────
    print("Plotting loss curves...")
    loss_curves_path = report.plot_loss_curves(results["loss_histories"])

    # ── 8. Generate PDF report ─────────────────
    print("Generating PDF report...")
    report.generate_pdf(results, aug_paths, pred_paths, loss_curves_path)

    print("\n✅ Pipeline complete!")
    print(f"   Report:      {config.REPORT_PATH}")
    print(f"   Checkpoints: {config.CHECKPOINTS_DIR}")
    print(f"   Logs:        {config.LOGS_DIR}")
    print(f"   Aug examples: {config.AUGMENTATION_EXAMPLES_DIR}")
    print(f"   Predictions:  {config.BEST_FOLD_PREDICTIONS_DIR}")


if __name__ == "__main__":
    main()
