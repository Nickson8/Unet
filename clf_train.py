"""
Classification training module: single-fold training and validation loops,
learning-rate scheduling, early stopping, and CSV logging for the
malignant/benign classifier.
"""

import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from tqdm import tqdm

import config
from clf_metrics import (
    compute_and_reset_clf,
    compute_per_class_metrics,
    create_clf_metrics,
    update_clf_metrics,
)
from clf_model import freeze_classifier_backbone, unfreeze_classifier_backbone
from train import EarlyStopping


# ═══════════════════════════════════════════════
# 1. Single-epoch routines
# ═══════════════════════════════════════════════

def train_clf_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    metrics: MetricCollection,
    device: str,
) -> Tuple[float, Dict[str, float]]:
    """
    Run one classification training epoch.

    Returns
    -------
    avg_loss : float
    metrics_dict : dict  {accuracy, precision, recall, f1, auroc, auprc}
    """
    model.train()
    running_loss = 0.0

    for inputs, targets in tqdm(loader, desc="  train", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device).long()

        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        update_clf_metrics(metrics, logits.detach(), targets)

    avg_loss = running_loss / len(loader.dataset)
    metrics_dict = compute_and_reset_clf(metrics)
    return avg_loss, metrics_dict


@torch.no_grad()
def validate_clf_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    metrics: MetricCollection,
    device: str,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Run one classification validation epoch.

    Returns
    -------
    avg_loss : float
    metrics_dict : dict  {accuracy, precision, recall, f1, auroc, auprc}
    all_preds : np.ndarray of predicted class indices
    all_targets : np.ndarray of true class indices
    """
    model.eval()
    running_loss = 0.0
    preds_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []

    for inputs, targets in tqdm(loader, desc="  val  ", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device).long()

        logits = model(inputs)
        loss = loss_fn(logits, targets)

        running_loss += loss.item() * inputs.size(0)
        update_clf_metrics(metrics, logits, targets)

        preds_list.append(logits.argmax(dim=1).cpu())
        targets_list.append(targets.cpu())

    avg_loss = running_loss / len(loader.dataset)
    metrics_dict = compute_and_reset_clf(metrics)

    all_preds = torch.cat(preds_list).numpy()
    all_targets = torch.cat(targets_list).numpy()

    return avg_loss, metrics_dict, all_preds, all_targets


# ═══════════════════════════════════════════════
# 2. Full fold training loop
# ═══════════════════════════════════════════════

def train_clf_fold(
    fold_idx: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    class_weights: torch.Tensor,
    variant: str = "independent",
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Train a classifier for one complete fold.

    Parameters
    ----------
    variant : ``"mask_aware"`` or ``"independent"`` — used for file naming.

    Returns
    -------
    best_metrics : dict — best validation metrics (by AUROC), including
                   per-class breakdown.
    epoch_logs   : list of dicts — per-epoch log records.
    """
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
    )

    # Metrics on device
    train_metrics = create_clf_metrics(device)
    val_metrics = create_clf_metrics(device)

    # Optimiser — initially only head params (backbone frozen)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.SCHEDULER_PATIENCE,
    )
    early_stop = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE, mode="max",
    )

    # Freeze backbone for warm-up
    freeze_classifier_backbone(model)

    # Logging
    log_dir = config.CLF_LOGS_DIR
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"clf_{variant}_fold_{fold_idx}.csv")
    log_file = open(log_path, "w", newline="")
    writer = csv.DictWriter(
        log_file,
        fieldnames=[
            "epoch", "train_loss", "val_loss",
            "accuracy", "auroc", "auprc", "f1", "lr",
        ],
    )
    writer.writeheader()

    best_auroc = -1.0
    best_metrics: Dict[str, float] = {}
    epoch_logs: List[Dict] = []
    ckpt_dir = config.CLF_CHECKPOINTS_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"best_clf_{variant}_fold_{fold_idx}.pth")

    for epoch in range(1, config.CLF_NUM_EPOCHS + 1):
        # Unfreeze backbone after warm-up and rebuild optimiser
        if epoch == config.FREEZE_ENCODER_EPOCHS + 1:
            unfreeze_classifier_backbone(model)
            # Discriminative LR: backbone gets LR/10
            classifier_head = model.get_classifier()
            head_param_ids = {id(p) for p in classifier_head.parameters()}
            backbone_params = [
                p for p in model.parameters()
                if id(p) not in head_param_ids and p.requires_grad
            ]
            head_params = [
                p for p in classifier_head.parameters() if p.requires_grad
            ]
            optimizer = AdamW(
                [
                    {"params": backbone_params, "lr": config.LR / 10},
                    {"params": head_params, "lr": config.LR},
                ],
                weight_decay=config.WEIGHT_DECAY,
            )
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5,
                patience=config.SCHEDULER_PATIENCE,
            )

        # Train
        train_loss, train_m = train_clf_one_epoch(
            model, train_loader, optimizer, loss_fn, train_metrics, device,
        )

        # Validate
        val_loss, val_m, val_preds, val_targets = validate_clf_one_epoch(
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
            "accuracy": f"{val_m['accuracy']:.4f}",
            "auroc": f"{val_m['auroc']:.4f}",
            "auprc": f"{val_m['auprc']:.4f}",
            "f1": f"{val_m['f1']:.4f}",
            "lr": f"{current_lr:.2e}",
        }
        writer.writerow(row)
        log_file.flush()
        epoch_logs.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
             **val_m}
        )

        print(
            f"  Fold {fold_idx} | Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Acc: {val_m['accuracy']:.4f} | AUROC: {val_m['auroc']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save best model (by AUROC)
        if val_m["auroc"] > best_auroc:
            best_auroc = val_m["auroc"]
            # Compute per-class breakdown for the best epoch
            per_class = compute_per_class_metrics(val_preds, val_targets)
            best_metrics = {**val_m, **per_class}
            torch.save(model.state_dict(), ckpt_path)

        # Early stopping
        if early_stop.step(val_m["auroc"]):
            print(
                f"  ⏹  Early stopping at epoch {epoch} (patience exhausted)."
            )
            break

    log_file.close()
    return best_metrics, epoch_logs
