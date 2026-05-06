"""
Training module: single-fold training and validation loops, loss
construction, learning-rate scheduling, early stopping, and CSV logging.
"""

import csv
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from tqdm import tqdm

import config
from metrics import compute_and_reset, update_metrics
from model import freeze_encoder, unfreeze_encoder


# ═══════════════════════════════════════════════
# 1. Combined loss
# ═══════════════════════════════════════════════

class CombinedLoss(nn.Module):
    """0.5 × DiceLoss + 0.5 × BCEWithLogitsLoss (from SMP)."""

    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(mode="binary", from_logits=True)
        self.bce = SoftBCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.dice(pred, target) + 0.5 * self.bce(pred, target)


# ═══════════════════════════════════════════════
# 2. Single-epoch routines
# ═══════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    metrics: MetricCollection,
    device: str,
) -> Tuple[float, Dict[str, float]]:
    """
    Run one training epoch.

    Returns
    -------
    avg_loss : float
    metrics_dict : dict  {dice, iou, accuracy, precision, recall}
    """
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = loss_fn(preds, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        update_metrics(metrics, preds.detach(), masks)

    avg_loss = running_loss / len(loader.dataset)
    metrics_dict = compute_and_reset(metrics)
    return avg_loss, metrics_dict


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    metrics: MetricCollection,
    device: str,
) -> Tuple[float, Dict[str, float]]:
    """
    Run one validation epoch (no gradients).

    Returns
    -------
    avg_loss : float
    metrics_dict : dict  {dice, iou, accuracy, precision, recall}
    """
    model.eval()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = loss_fn(preds, masks)

        running_loss += loss.item() * images.size(0)
        update_metrics(metrics, preds, masks)

    avg_loss = running_loss / len(loader.dataset)
    metrics_dict = compute_and_reset(metrics)
    return avg_loss, metrics_dict


# ═══════════════════════════════════════════════
# 3. Early stopping helper
# ═══════════════════════════════════════════════

class EarlyStopping:
    """Stop training when the monitored metric stops improving."""

    def __init__(self, patience: int = 15, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.counter = 0
        self.triggered = False

    def step(self, value: float) -> bool:
        """Return True if training should stop."""
        if self.best is None:
            self.best = value
            return False

        improved = (
            value > self.best if self.mode == "max" else value < self.best
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                return True
        return False


# ═══════════════════════════════════════════════
# 4. Full fold training loop
# ═══════════════════════════════════════════════

def train_fold(
    fold_idx: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Train a model for one complete fold: N epochs with early stopping.

    Returns
    -------
    best_metrics : dict  — best validation metrics (by Dice)
    epoch_logs   : list of dicts — per-epoch log records
    """
    model = model.to(device)
    loss_fn = CombinedLoss().to(device)

    # Metrics on device
    from metrics import create_metrics

    train_metrics = create_metrics(device)
    val_metrics = create_metrics(device)

    # Optimiser — initially only decoder params (encoder frozen)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.SCHEDULER_PATIENCE
    )
    early_stop = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE, mode="max"
    )

    # Freeze encoder for warm-up
    freeze_encoder(model)

    # Logging
    log_path = os.path.join(config.LOGS_DIR, f"fold_{fold_idx}.csv")
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    log_file = open(log_path, "w", newline="")
    writer = csv.DictWriter(
        log_file,
        fieldnames=[
            "epoch", "train_loss", "val_loss",
            "dice", "iou", "accuracy", "precision", "recall", "lr",
        ],
    )
    writer.writeheader()

    best_dice = -1.0
    best_metrics: Dict[str, float] = {}
    epoch_logs: List[Dict] = []
    ckpt_path = os.path.join(
        config.CHECKPOINTS_DIR, f"best_model_fold_{fold_idx}.pth"
    )
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Unfreeze encoder after warm-up and rebuild optimiser
        if epoch == config.FREEZE_ENCODER_EPOCHS + 1:
            unfreeze_encoder(model)
            optimizer = AdamW(
                [
                    {"params": model.encoder.parameters(), "lr": config.LR / 10},
                    {"params": model.decoder.parameters(), "lr": config.LR},
                    {"params": model.segmentation_head.parameters(), "lr": config.LR},
                ],
                weight_decay=config.WEIGHT_DECAY,
            )
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5,
                patience=config.SCHEDULER_PATIENCE,
            )

        # Train
        train_loss, train_m = train_one_epoch(
            model, train_loader, optimizer, loss_fn, train_metrics, device,
        )

        # Validate
        val_loss, val_m = validate_one_epoch(
            model, val_loader, loss_fn, val_metrics, device,
        )

        # LR scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "dice": f"{val_m['dice']:.4f}",
            "iou": f"{val_m['iou']:.4f}",
            "accuracy": f"{val_m['accuracy']:.4f}",
            "precision": f"{val_m['precision']:.4f}",
            "recall": f"{val_m['recall']:.4f}",
            "lr": f"{current_lr:.2e}",
        }
        writer.writerow(row)
        log_file.flush()
        epoch_logs.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_m}
        )

        print(
            f"  Fold {fold_idx} | Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Dice: {val_m['dice']:.4f} | IoU: {val_m['iou']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save best model (by Dice)
        if val_m["dice"] > best_dice:
            best_dice = val_m["dice"]
            best_metrics = val_m.copy()
            torch.save(model.state_dict(), ckpt_path)

        # Early stopping
        if early_stop.step(val_m["dice"]):
            print(f"  ⏹  Early stopping at epoch {epoch} (patience exhausted).")
            break

    log_file.close()
    return best_metrics, epoch_logs
