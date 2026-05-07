"""
Central configuration for the U-Net dental lesion segmentation pipeline.
All hyperparameters and paths are defined here as a single source of truth.
"""

import os
import torch

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
DATA_DIR = os.environ.get("UNET_DATA_DIR", "./dataset")
IMAGE_EXTENSIONS = (".png", ".jpeg", ".jpg")

# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
IMAGE_SIZE = 512  # All images resized to IMAGE_SIZE x IMAGE_SIZE

# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
ENCODER_NAME = "tu-convnextv2_nano"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 3   # RGB (alpha channel dropped during loading)
NUM_CLASSES = 1   # Binary segmentation (lesion vs. background)

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
BATCH_SIZE = 2
NUM_EPOCHS = 2
LR = 1e-4
WEIGHT_DECAY = 1e-4
FREEZE_ENCODER_EPOCHS = 5   # Freeze encoder for first N epochs
SCHEDULER_PATIENCE = 7      # ReduceLROnPlateau patience
EARLY_STOPPING_PATIENCE = 15

# ──────────────────────────────────────────────
# Cross-validation
# ──────────────────────────────────────────────
NUM_FOLDS = 5
SEED = 42

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
# Classification (Stage 2)
# ──────────────────────────────────────────────
CLF_NUM_EPOCHS = 2  # May differ from segmentation epochs

# ──────────────────────────────────────────────
# Output directories (created at runtime)
# ──────────────────────────────────────────────
OUTPUT_DIR = "./outputs"
AUGMENTATION_EXAMPLES_DIR = os.path.join(OUTPUT_DIR, "augmentation_examples")
BEST_FOLD_PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "best_fold_predictions")
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
CLF_CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "clf_checkpoints")
CLF_LOGS_DIR = os.path.join(OUTPUT_DIR, "clf_logs")
REPORT_PATH = os.path.join(OUTPUT_DIR, "final_report.pdf")

# ──────────────────────────────────────────────
# ImageNet normalization (used by SMP encoders)
# ──────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
