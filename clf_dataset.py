"""
Classification dataset module: label extraction from filename prefix,
PyTorch Dataset for mask-aware and independent classification modes,
augmentation pipelines, and class-weight computation.
"""

import os
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config
from dataset import geojson_to_mask


# ═══════════════════════════════════════════════
# 1. Label extraction
# ═══════════════════════════════════════════════

def extract_label(image_path: str) -> int:
    """
    Return 1 (malignant) if the filename starts with ``M_``, else 0 (benign).
    """
    basename = os.path.basename(image_path)
    return 1 if basename.startswith("M_") else 0


# ═══════════════════════════════════════════════
# 2. Class weights
# ═══════════════════════════════════════════════

def compute_class_weights(pairs: List[Tuple[str, str]]) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    Returns a float tensor of shape (2,) — [weight_benign, weight_malignant].
    """
    labels = [extract_label(p[0]) for p in pairs]
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    total = len(labels)
    weights = total / (2.0 * counts)
    return torch.FloatTensor(weights)


# ═══════════════════════════════════════════════
# 3. Augmentation pipelines
# ═══════════════════════════════════════════════

def get_clf_train_transforms() -> A.Compose:
    """Heavy augmentation pipeline for classification training."""
    return A.Compose(
        [
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            # --- Spatial ---
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
            # --- Pixel-level ---
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(0.03, 0.08),
                hole_width_range=(0.03, 0.08),
                fill=0,
                p=0.3,
            ),
            # --- Normalise + to tensor ---
            A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_clf_val_transforms() -> A.Compose:
    """Minimal transforms for classification validation."""
    return A.Compose(
        [
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ToTensorV2(),
        ]
    )


# ═══════════════════════════════════════════════
# 4. PyTorch Dataset
# ═══════════════════════════════════════════════

class ClassificationDataset(Dataset):
    """
    Dataset for malignant/benign classification.

    Modes
    -----
    ``"mask_aware"``  : returns (4-channel tensor [RGB + mask], label)
    ``"independent"`` : returns (3-channel tensor [RGB], label)
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        transform: A.Compose | None = None,
        mode: str = "independent",
    ):
        self.pairs = pairs
        self.transform = transform
        self.mode = mode

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        image_path, geojson_path = self.pairs[idx]
        label = extract_label(image_path)

        # Load image as RGB
        image = np.array(Image.open(image_path).convert("RGB"))[:, :, :3]

        # Generate mask (needed for mask_aware; also for joint spatial transforms)
        mask = geojson_to_mask(geojson_path, image.shape[:2])

        # Apply augmentations jointly to image + mask
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]    # (C, H, W) float tensor
            mask = transformed["mask"]      # (H, W) tensor
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask)

        # Build input tensor
        if self.mode == "mask_aware":
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)        # (1, H, W)
            input_tensor = torch.cat([image, mask.float()], dim=0)  # (4, H, W)
        else:
            input_tensor = image                # (3, H, W)

        return input_tensor, label


# ═══════════════════════════════════════════════
# 5. DataLoader factory
# ═══════════════════════════════════════════════

def get_clf_fold_dataloaders(
    all_pairs: List[Tuple[str, str]],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    mode: str = "independent",
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders for a single fold.
    """
    train_pairs = [all_pairs[i] for i in train_indices]
    val_pairs = [all_pairs[i] for i in val_indices]

    train_ds = ClassificationDataset(
        train_pairs, transform=get_clf_train_transforms(), mode=mode,
    )
    val_ds = ClassificationDataset(
        val_pairs, transform=get_clf_val_transforms(), mode=mode,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
