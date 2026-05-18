"""Pretrained vision-model backbones adapted for CFDAC inputs.

We compare cnn2d (the bespoke 2-D CNN trained from scratch on CFDAC)
against five general-purpose ImageNet-pretrained vision models.  Each
model is loaded with ImageNet weights, its first conv (or patch
embedding) is replaced to accept ``n_channels`` from the chosen CFDAC
variant, and the classifier head is replaced with a fresh n_out
Linear.

The five backbones cover the standard transfer-learning archetypes:

  resnet50            classic deep CNN, 25.6 M params
  efficientnet_b0     small efficient CNN, 5.3 M params
  convnext_tiny       modern convnet, 28.6 M params
  swin_t              hierarchical transformer, 28.3 M params
  vit_b_16            vanilla vision transformer, 86.6 M params

Inputs are 128×128 CFDAC tensors which the wrapper upsamples to 224×224
on the fly (the size every torchvision model is pretrained on).  When
adapting from 3 ImageNet channels to a different ``n_channels``, the
new first conv is initialised from the channel-mean of the pretrained
weights, then repeated/clipped to the target channel count.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

try:
    import torchvision.models as tvm
    from torchvision.models import (
        ResNet50_Weights, EfficientNet_B0_Weights, ViT_B_16_Weights,
        ConvNeXt_Tiny_Weights, Swin_T_Weights,
    )
    _HAS_TV = True
except Exception:
    _HAS_TV = False


VISION_BACKBONES = {
    "resnet50":        lambda w: tvm.resnet50(weights=w),
    "efficientnet_b0": lambda w: tvm.efficientnet_b0(weights=w),
    "convnext_tiny":   lambda w: tvm.convnext_tiny(weights=w),
    "swin_t":          lambda w: tvm.swin_t(weights=w),
    "vit_b_16":        lambda w: tvm.vit_b_16(weights=w),
}


def _adapt_first_conv(old: nn.Conv2d, n_channels: int) -> nn.Conv2d:
    """Build a new Conv2d that accepts ``n_channels`` and initialise
    its weights from the channel-mean of ``old`` (the pretrained 3-ch
    first conv), repeated to fill the new channel count."""
    new = nn.Conv2d(
        n_channels, old.out_channels,
        kernel_size=old.kernel_size, stride=old.stride,
        padding=old.padding, bias=old.bias is not None,
    )
    with torch.no_grad():
        # Source first conv expects 3 channels (ImageNet RGB).  Take the
        # per-out-channel mean across input channels then tile to whatever
        # n_channels we need.  Scale by 3/n_channels so the receptive
        # response stays roughly the same magnitude.
        ch_mean = old.weight.mean(dim=1, keepdim=True)        # (out, 1, kH, kW)
        tiled = ch_mean.repeat(1, n_channels, 1, 1) * (3.0 / max(n_channels, 1))
        new.weight.copy_(tiled)
        if old.bias is not None and new.bias is not None:
            new.bias.copy_(old.bias)
    return new


def _adapt_first_linear(old: nn.Linear, n_channels: int) -> nn.Linear:
    """For ViT, the patch embedding is actually a Conv2d on flatten
    patches.  But for Swin and ConvNeXt the stem starts with a Conv2d
    we can adapt directly.  This is a fallback helper."""
    raise NotImplementedError


def _replace_head(model: nn.Module, name: str, n_out: int) -> None:
    """Swap the classifier head for a fresh n_out Linear."""
    if name == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, n_out)
    elif name == "efficientnet_b0":
        # classifier is Sequential(Dropout, Linear)
        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, n_out)
    elif name == "convnext_tiny":
        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, n_out)
    elif name == "swin_t":
        model.head = nn.Linear(model.head.in_features, n_out)
    elif name == "vit_b_16":
        # heads is Sequential(Linear). Replace it.
        in_feats = model.heads.head.in_features
        model.heads.head = nn.Linear(in_feats, n_out)
    else:
        raise ValueError(name)


def _replace_first_conv(model: nn.Module, name: str, n_channels: int) -> None:
    """Swap the model's first convolution to accept ``n_channels``."""
    if name == "resnet50":
        model.conv1 = _adapt_first_conv(model.conv1, n_channels)
    elif name == "efficientnet_b0":
        # features[0] is Conv2dNormActivation; its [0] is the Conv2d.
        old = model.features[0][0]
        model.features[0][0] = _adapt_first_conv(old, n_channels)
    elif name == "convnext_tiny":
        # features[0][0] is the stem Conv2d.
        old = model.features[0][0]
        model.features[0][0] = _adapt_first_conv(old, n_channels)
    elif name == "swin_t":
        # features[0][0] is the patch embedding Conv2d.
        old = model.features[0][0]
        model.features[0][0] = _adapt_first_conv(old, n_channels)
    elif name == "vit_b_16":
        # conv_proj is the patch embedding Conv2d.
        model.conv_proj = _adapt_first_conv(model.conv_proj, n_channels)
    else:
        raise ValueError(name)


class VisionBackbone(nn.Module):
    """Thin wrapper around a torchvision model that handles input
    upsampling, the bounded-regression flag, and lets the outer code
    pretend every backbone has the same forward signature."""

    def __init__(self, name: str, n_channels: int, n_out: int,
                  regression: bool = False, bounded_output: bool = True,
                  pretrained: bool = True, target_size: int = 224):
        super().__init__()
        if not _HAS_TV:
            raise ImportError("torchvision required for vision backbones")
        if name not in VISION_BACKBONES:
            raise ValueError(f"unknown vision backbone {name!r}")
        ctor = VISION_BACKBONES[name]
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = ctor(weights)
        _replace_first_conv(backbone, name, n_channels)
        _replace_head(backbone, name, n_out)
        self.backbone = backbone
        self.name = name
        self.regression = regression
        self.bounded_output = bounded_output
        self.target_size = int(target_size)
        # ViT requires fixed size; the others accept any divisible size
        # but the pretrained features are aligned at 224 so we keep it.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — CFDAC is typically 128x128.
        if x.ndim != 4:
            raise ValueError(f"VisionBackbone expects (B, C, H, W), got {x.shape}")
        if x.shape[-1] != self.target_size or x.shape[-2] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode="bilinear",
                                  align_corners=False)
        out = self.backbone(x)
        if self.regression and self.bounded_output:
            out = torch.sigmoid(out)
        return out


def is_vision_backbone(name: str) -> bool:
    return name in VISION_BACKBONES
