"""
Report module: generates augmentation examples, segmentation overlays,
loss-curve plots, and assembles everything into a PDF report.
"""

import os
import random
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from fpdf import FPDF
from PIL import Image

import config
from dataset import (
    DentalDataset,
    geojson_to_mask,
    get_train_transforms,
    get_val_transforms,
)
from model import create_model

matplotlib.use("Agg")  # non-interactive backend


# ═══════════════════════════════════════════════
# 1. Augmentation examples
# ═══════════════════════════════════════════════

def save_augmentation_examples(
    pairs: List[Tuple[str, str]],
    n_examples: int = 5,
    seed: int = config.SEED,
) -> List[str]:
    """
    Pick *n_examples* random images, apply training augmentation, and
    save side-by-side comparisons (original | augmented) to disk.

    Returns a list of saved image paths.
    """
    os.makedirs(config.AUGMENTATION_EXAMPLES_DIR, exist_ok=True)
    rng = random.Random(seed)
    chosen = rng.sample(pairs, min(n_examples, len(pairs)))

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # Augmentation without normalise/tensor (for visualisation)
    vis_transform = A.Compose(
        [
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent=(-0.1, 0.1),
                scale=(0.85, 1.15),
                rotate=(-30, 30),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=0.7,
            ),
            A.ElasticTransform(alpha=120, sigma=6, p=0.3),
            A.GridDistortion(p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
        ]
    )

    saved_paths: List[str] = []
    for i, (img_path, geo_path) in enumerate(chosen):
        image = np.array(Image.open(img_path).convert("RGB"))[:, :, :3]
        mask = geojson_to_mask(geo_path, image.shape[:2])

        # Resize original for consistent comparison
        orig_resized = cv2.resize(
            image, (config.IMAGE_SIZE, config.IMAGE_SIZE)
        )
        mask_resized = cv2.resize(
            mask, (config.IMAGE_SIZE, config.IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

        # Apply augmentation
        augmented = vis_transform(image=image, mask=mask)
        aug_img = augmented["image"]
        aug_mask = augmented["mask"]

        # Create figure: 2×2 grid (orig, orig+mask, aug, aug+mask)
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(
            f"Augmentation Example {i + 1}", fontsize=14, fontweight="bold"
        )

        axes[0, 0].imshow(orig_resized)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(orig_resized)
        axes[0, 1].imshow(mask_resized, alpha=0.4, cmap="Reds")
        axes[0, 1].set_title("Original + Mask")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(aug_img)
        axes[1, 0].set_title("Augmented Image")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(aug_img)
        axes[1, 1].imshow(aug_mask, alpha=0.4, cmap="Reds")
        axes[1, 1].set_title("Augmented + Mask")
        axes[1, 1].axis("off")

        plt.tight_layout()
        save_path = os.path.join(
            config.AUGMENTATION_EXAMPLES_DIR, f"aug_example_{i + 1}.png"
        )
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(save_path)

    print(f"  Saved {len(saved_paths)} augmentation examples → "
          f"{config.AUGMENTATION_EXAMPLES_DIR}")
    return saved_paths


# ═══════════════════════════════════════════════
# 2. Segmentation prediction overlays
# ═══════════════════════════════════════════════

def save_prediction_examples(
    pairs: List[Tuple[str, str]],
    val_indices: np.ndarray,
    best_fold_idx: int,
    n_examples: int = 5,
) -> List[str]:
    """
    Load the best-fold checkpoint, run inference on *n_examples* validation
    images, and save overlays (image + GT contour + prediction mask).

    Returns a list of saved image paths.
    """
    os.makedirs(config.BEST_FOLD_PREDICTIONS_DIR, exist_ok=True)

    # Load best model
    model = create_model()
    ckpt_path = os.path.join(
        config.CHECKPOINTS_DIR, f"best_model_fold_{best_fold_idx}.pth"
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE, weights_only=True))
    model = model.to(config.DEVICE)
    model.eval()

    val_transform = get_val_transforms()

    val_pairs = [pairs[i] for i in val_indices]
    rng = random.Random(config.SEED)
    chosen = rng.sample(val_pairs, min(n_examples, len(val_pairs)))

    saved_paths: List[str] = []
    with torch.no_grad():
        for i, (img_path, geo_path) in enumerate(chosen):
            # Load raw image and mask
            raw_image = np.array(Image.open(img_path).convert("RGB"))[:, :, :3]
            gt_mask = geojson_to_mask(geo_path, raw_image.shape[:2])

            # Prepare for model
            transformed = val_transform(image=raw_image, mask=gt_mask)
            input_tensor = transformed["image"].unsqueeze(0).to(config.DEVICE)

            # Predict
            logits = model(input_tensor)
            pred_mask = (torch.sigmoid(logits) > 0.5).cpu().numpy()[0, 0]

            # Resize raw image for display
            display_img = cv2.resize(
                raw_image, (config.IMAGE_SIZE, config.IMAGE_SIZE)
            )
            gt_resized = cv2.resize(
                gt_mask, (config.IMAGE_SIZE, config.IMAGE_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )

            # Build figure: original, GT, prediction, overlay
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle(
                f"Prediction Example {i + 1} (Fold {best_fold_idx})",
                fontsize=14, fontweight="bold",
            )

            axes[0].imshow(display_img)
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(gt_resized, cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(pred_mask, cmap="gray")
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            # Overlay: GT contour (green) + prediction (red, semi-transparent)
            axes[3].imshow(display_img)
            axes[3].imshow(pred_mask, alpha=0.35, cmap="Reds")
            # Draw GT contour
            gt_contours, _ = cv2.findContours(
                gt_resized.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            contour_overlay = np.zeros(
                (*gt_resized.shape, 4), dtype=np.float32
            )
            cv2.drawContours(
                contour_overlay, gt_contours, -1,
                (0, 1, 0, 1), thickness=2,
            )
            axes[3].imshow(contour_overlay)
            axes[3].set_title("Overlay (Red=Pred, Green=GT)")
            axes[3].axis("off")

            plt.tight_layout()
            save_path = os.path.join(
                config.BEST_FOLD_PREDICTIONS_DIR,
                f"pred_example_{i + 1}.png",
            )
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(save_path)

    print(f"  Saved {len(saved_paths)} prediction examples → "
          f"{config.BEST_FOLD_PREDICTIONS_DIR}")
    return saved_paths


# ═══════════════════════════════════════════════
# 3. Loss curves
# ═══════════════════════════════════════════════

def plot_loss_curves(loss_histories: List[List[Dict]]) -> str:
    """
    Plot train/val loss per epoch for each fold in a grid.

    Returns the path to the saved figure.
    """
    n_folds = len(loss_histories)
    fig, axes = plt.subplots(1, n_folds, figsize=(5 * n_folds, 4), sharey=True)
    if n_folds == 1:
        axes = [axes]

    for idx, logs in enumerate(loss_histories):
        epochs = [r["epoch"] for r in logs]
        train_losses = [r["train_loss"] for r in logs]
        val_losses = [r["val_loss"] for r in logs]

        axes[idx].plot(epochs, train_losses, label="Train", linewidth=1.5)
        axes[idx].plot(epochs, val_losses, label="Val", linewidth=1.5)
        axes[idx].set_title(f"Fold {idx + 1}", fontweight="bold")
        axes[idx].set_xlabel("Epoch")
        if idx == 0:
            axes[idx].set_ylabel("Loss")
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)

    fig.suptitle("Training & Validation Loss Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, "loss_curves.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ═══════════════════════════════════════════════
# 4. PDF report
# ═══════════════════════════════════════════════

class ReportPDF(FPDF):
    """Custom FPDF subclass with header/footer."""

    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, "U-Net Dental Lesion Segmentation - Report", align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def generate_pdf(
    results: Dict[str, Any],
    aug_paths: List[str],
    pred_paths: List[str],
    loss_curves_path: str,
) -> str:
    """
    Assemble the final PDF report.

    Sections:
        1. Dataset summary (total, train/val counts)
        2. Metrics table (mean ± SE, per-fold breakdown)
        3. Loss curves
        4. Augmentation examples
        5. Prediction examples

    Returns the path to the saved PDF.
    """
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Title page ─────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.ln(40)
    pdf.cell(0, 15, "U-Net Dental Lesion Segmentation", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, "5-Fold Cross-Validation Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Encoder: {config.ENCODER_NAME}  |  "
                    f"Image Size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}  |  "
                    f"Batch Size: {config.BATCH_SIZE}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"LR: {config.LR}  |  Epochs: {config.NUM_EPOCHS}  |  "
                    f"Loss: Dice + BCE", align="C", new_x="LMARGIN", new_y="NEXT")

    # ── Section 1: Dataset Summary ─────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "1. Dataset Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    info = results["dataset_info"]
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Total images: {info['total']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Training images per fold: ~{info['train_per_fold']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Validation images per fold: ~{info['val_per_fold']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Number of folds: {config.NUM_FOLDS}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── Section 2: Metrics ─────────────────────
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "2. Cross-Validation Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Mean ± SE table
    pdf.set_font("Helvetica", "B", 10)
    col_w = 35
    metric_names = ["dice", "iou", "accuracy", "precision", "recall"]

    # Header row
    pdf.cell(col_w, 8, "Metric", border=1, align="C")
    pdf.cell(col_w, 8, "Mean", border=1, align="C")
    pdf.cell(col_w, 8, "Std Error", border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    for name in metric_names:
        mean_val = results["mean_metrics"][name]
        se_val = results["std_error_metrics"][name]
        pdf.cell(col_w, 7, name.capitalize(), border=1, align="C")
        pdf.cell(col_w, 7, f"{mean_val:.4f}", border=1, align="C")
        pdf.cell(col_w, 7, f"+/- {se_val:.4f}", border=1, align="C")
        pdf.ln()

    pdf.ln(6)

    # Per-fold breakdown
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Per-Fold Breakdown:", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 9)
    fold_col_w = 22
    pdf.cell(fold_col_w, 7, "Fold", border=1, align="C")
    for name in metric_names:
        pdf.cell(fold_col_w + 6, 7, name.capitalize(), border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    for i, fm in enumerate(results["fold_metrics"]):
        is_best = (i + 1 == results["best_fold_idx"])
        label = f"{i + 1} {'*' if is_best else ''}"
        pdf.cell(fold_col_w, 7, label, border=1, align="C")
        for name in metric_names:
            pdf.cell(fold_col_w + 6, 7, f"{fm[name]:.4f}", border=1, align="C")
        pdf.ln()

    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 6, f"* Best fold (highest Dice): Fold {results['best_fold_idx']}", new_x="LMARGIN", new_y="NEXT")

    # ── Section 3: Loss Curves ─────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "3. Loss Curves", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    if os.path.exists(loss_curves_path):
        # Fit to page width
        pdf.image(loss_curves_path, x=10, w=pdf.w - 20)

    # ── Section 4: Augmentation Examples ───────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "4. Data Augmentation Examples", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Each example shows: Original | Original+Mask | Augmented | Augmented+Mask",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    for path in aug_paths:
        if os.path.exists(path):
            if pdf.get_y() > pdf.h - 120:
                pdf.add_page()
            pdf.image(path, x=10, w=pdf.w - 20)
            pdf.ln(4)

    # ── Section 5: Prediction Examples ─────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "5. Segmentation Predictions (Best Fold)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Each example shows: Original | Ground Truth | Prediction | Overlay (Red=Pred, Green=GT)",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    for path in pred_paths:
        if os.path.exists(path):
            if pdf.get_y() > pdf.h - 80:
                pdf.add_page()
            pdf.image(path, x=10, w=pdf.w - 20)
            pdf.ln(4)

    # ── Save ───────────────────────────────────
    os.makedirs(os.path.dirname(config.REPORT_PATH), exist_ok=True)
    pdf.output(config.REPORT_PATH)
    print(f"\n  📄 Report saved → {config.REPORT_PATH}")
    return config.REPORT_PATH
