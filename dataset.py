"""
Dataset module: data discovery, GeoJSON→mask conversion, PyTorch Dataset,
and Albumentations augmentation pipelines.
"""

import json
import os
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from skimage.draw import polygon as ski_polygon
from torch.utils.data import DataLoader, Dataset

import config


# ═══════════════════════════════════════════════
# 1. Data discovery
# ═══════════════════════════════════════════════

def discover_image_mask_pairs(data_dir: str) -> List[Tuple[str, str]]:
    """
    Scan *data_dir* for image files (.png / .jpeg) that have a matching
    .geojson annotation with the same basename.

    Returns
    -------
    pairs : list of (image_path, geojson_path)
    """
    all_files = os.listdir(data_dir)
    image_files = sorted(
        f for f in all_files if f.lower().endswith(config.IMAGE_EXTENSIONS)
    )
    geojson_files = {f for f in all_files if f.endswith(".geojson")}

    pairs: List[Tuple[str, str]] = []
    for img_file in image_files:
        base = os.path.splitext(img_file)[0]
        geojson_name = f"{base}.geojson"
        if geojson_name in geojson_files:
            pairs.append(
                (
                    os.path.join(data_dir, img_file),
                    os.path.join(data_dir, geojson_name),
                )
            )
        else:
            print(f"[WARN] No GeoJSON found for {img_file}. Skipping.")
    return pairs


# ═══════════════════════════════════════════════
# 2. GeoJSON → binary mask
# ═══════════════════════════════════════════════

def geojson_to_mask(geojson_path: str, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Parse a GeoJSON file and rasterise its polygons into a binary mask
    of shape *image_shape* (H, W).  Handles both Polygon and MultiPolygon
    geometry types.  Coordinates are assumed to be in pixel space.

    Parameters
    ----------
    geojson_path : str
        Path to the .geojson file.
    image_shape : tuple of (height, width)
        Shape of the corresponding image (used for the output mask).

    Returns
    -------
    mask : np.ndarray, dtype uint8, shape (H, W), values in {0, 1}
    """
    with open(geojson_path, "r") as f:
        geojson_data = json.load(f)

    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for feature in geojson_data.get("features", [geojson_data]):
        geometry = feature.get("geometry", feature)
        geom_type = geometry.get("type", "")

        if geom_type == "Polygon":
            polygon_rings = [geometry["coordinates"][0]]
        elif geom_type == "MultiPolygon":
            polygon_rings = [poly[0] for poly in geometry["coordinates"]]
        else:
            continue

        for ring in polygon_rings:
            poly_array = np.array(ring)
            if poly_array.size == 0:
                continue
            # GeoJSON uses [x, y] → skimage.draw.polygon expects (row, col)
            rr, cc = ski_polygon(poly_array[:, 1], poly_array[:, 0], mask.shape)
            mask[rr, cc] = 1

    return mask


# ═══════════════════════════════════════════════
# 3. Augmentation pipelines
# ═══════════════════════════════════════════════

def get_train_transforms() -> A.Compose:
    """Heavy augmentation pipeline for training."""
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


def get_val_transforms() -> A.Compose:
    """Minimal transforms for validation (resize + normalise only)."""
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

class DentalDataset(Dataset):
    """
    Lazy-loading dataset: each call to __getitem__ reads the image from
    disk, generates the mask from the GeoJSON, and applies augmentations.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        transform: A.Compose | None = None,
    ):
        """
        Parameters
        ----------
        pairs : list of (image_path, geojson_path)
        transform : albumentations Compose pipeline (or None)
        """
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        image_path, geojson_path = self.pairs[idx]

        # Load image as RGB, drop alpha if present
        image = np.array(Image.open(image_path).convert("RGB"))[:, :, :3]

        # Generate binary mask from GeoJSON
        mask = geojson_to_mask(geojson_path, image.shape[:2])

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]       # (C, H, W) float tensor
            mask = transformed["mask"]         # (H, W) uint8/float tensor

        # Ensure mask has a channel dimension: (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask.float()


# ═══════════════════════════════════════════════
# 5. DataLoader factory
# ═══════════════════════════════════════════════

def get_fold_dataloaders(
    all_pairs: List[Tuple[str, str]],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders for a single fold.
    """
    train_pairs = [all_pairs[i] for i in train_indices]
    val_pairs = [all_pairs[i] for i in val_indices]

    train_ds = DentalDataset(train_pairs, transform=get_train_transforms())
    val_ds = DentalDataset(val_pairs, transform=get_val_transforms())

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
