"""
Model module: U-Net creation via segmentation_models_pytorch, plus
helpers for freezing/unfreezing the encoder.
"""

import segmentation_models_pytorch as smp
import torch.nn as nn
import torch

import config


def create_model() -> nn.Module:
    """
    Build an SMP U-Net with the encoder and weights specified in config.

    Returns a model that outputs **raw logits** (no final activation).
    """
    if(config.ENCODER_WEIGHTS != "radimagenet"):
        model = smp.Unet(
            encoder_name=config.ENCODER_NAME,
            encoder_weights=config.ENCODER_WEIGHTS,
            in_channels=config.IN_CHANNELS,
            classes=config.NUM_CLASSES,
            activation=None,  # raw logits — loss functions apply sigmoid
        )
        return model
    else:
        # 1. Create the model (e.g., Unet with ResNet50 encoder)
        model = smp.Unet(
            encoder_name=config.ENCODER_NAME,
            encoder_weights=None,
            in_channels=config.IN_CHANNELS,
            classes=config.NUM_CLASSES,
            activation=None,  # raw logits — loss functions apply sigmoid
        )

        # 2. Load the downloaded RadImageNet weights
        # Assuming you downloaded 'ResNet50.pt'
        radimagenet_data = torch.load(f"{config.ENCODER_NAME}.pt", map_location="cpu")

        # RadImageNet files often contain the full model; extract the state_dict
        if isinstance(radimagenet_data, torch.nn.Module):
            rad_state_dict = radimagenet_data.state_dict()
        else:
            rad_state_dict = radimagenet_data

        # 3. Filter weights to match the encoder
        # SMP encoders usually have a 'model' or 'layer' prefix that might differ
        encoder_state_dict = model.encoder.state_dict()
        new_state_dict = {}

        for key, value in rad_state_dict.items():
            # We only want layers that exist in the SMP encoder
            # This often requires stripping 'fc' (fully connected) layers used for classification
            if key in encoder_state_dict and value.size() == encoder_state_dict[key].size():
                new_state_dict[key] = value

        # 4. Load the weights into the encoder
        model.encoder.load_state_dict(new_state_dict, strict=False)
        print(f"Successfully loaded {len(new_state_dict)} tensors into the encoder.")

        return model


def freeze_encoder(model: nn.Module) -> None:
    """Freeze all encoder parameters (no gradient updates)."""
    for param in model.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model: nn.Module) -> None:
    """Unfreeze all encoder parameters."""
    for param in model.encoder.parameters():
        param.requires_grad = True
