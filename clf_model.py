"""
Classification model module: creates a timm-based image classifier for
malignant/benign classification. Supports 3-channel (independent) and
4-channel (mask-aware) input modes.
"""

import timm
import torch
import torch.nn as nn

import config


# ═══════════════════════════════════════════════
# 1. Internal helpers
# ═══════════════════════════════════════════════

def _get_first_conv_info(model: nn.Module):
    """
    Find the first Conv2d layer in the model.

    Returns (dotted_name, module).
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            return name, module
    raise ValueError("No Conv2d layer found in model.")


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """Replace a submodule identified by its dotted name path."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = (
            parent[int(part)] if part.isdigit() else getattr(parent, part)
        )
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def _replace_first_conv(model: nn.Module, new_in_channels: int) -> None:
    """
    Replace the first Conv2d in *model* with one accepting
    *new_in_channels* inputs.  Pretrained weights for the original
    channels are copied; extra channels are zero-initialised.
    """
    name, old_conv = _get_first_conv_info(model)
    new_conv = nn.Conv2d(
        new_in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )
    with torch.no_grad():
        # Copy pretrained weights for the first 3 (RGB) channels
        new_conv.weight[:, :3] = old_conv.weight[:, :3]
        # Zero-initialise the extra (mask) channel(s)
        new_conv.weight[:, 3:] = 0.0
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    _set_module_by_name(model, name, new_conv)


# ═══════════════════════════════════════════════
# 2. Public API
# ═══════════════════════════════════════════════

def create_classifier(mode: str = "independent") -> nn.Module:
    """
    Build a timm classifier using the same backbone as the U-Net encoder.

    Parameters
    ----------
    mode : ``"independent"`` (3-channel RGB) or ``"mask_aware"``
           (4-channel RGB + mask).

    Returns
    -------
    nn.Module — classifier producing (N, 2) logits.
    """
    model = timm.create_model(
        config.ENCODER_NAME[3:],
        pretrained=True,
        num_classes=2,
    )

    if mode == "mask_aware":
        _replace_first_conv(model, new_in_channels=4)

    return model


def freeze_classifier_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classification head."""
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the classifier head
    classifier = model.get_classifier()
    for param in classifier.parameters():
        param.requires_grad = True


def unfreeze_classifier_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True
