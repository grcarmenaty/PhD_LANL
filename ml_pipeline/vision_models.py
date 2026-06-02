"""Pretrained vision-model backbones adapted for CFDAC inputs (timm).

We compare cnn2d (the bespoke 2-D CNN trained from scratch on CFDAC)
against general-purpose ImageNet-pretrained vision models. This build
uses **timm** rather than torchvision, because the official torchvision
weight host (``download.pytorch.org``) and the HuggingFace hub are both
blocked by this environment's network policy, whereas timm's legacy
weights are mirrored as **GitHub release assets**, which are reachable.

For the synth-to-real study we use the top-3 backbones by cross-domain
macro-F1 (see REPORT_vision.md), mapped to the closest timm variant whose
ImageNet-1k weights are hosted on a GitHub release:

  friendly name    timm variant                              weights host
  resnet50         resnet50.a1_in1k                          github (rwightman)
  convnext_tiny    convnext_tiny_hnf.a2h_in1k                github (rwightman)
  vit_b_16         vit_base_patch16_224.orig_in21k_ft_in1k   github (rwightman)

timm natively adapts the first conv / patch embed to an arbitrary
``n_channels`` (``in_chans``) by summing/averaging the pretrained RGB
kernel, and builds a fresh classifier head (``num_classes``). We keep the
backbone as a headless feature extractor (``num_classes=0``) and attach
our own ``nn.Linear`` head so the outer training loop's linear-probe →
fine-tune schedule (freeze ``mdl.backbone``, train the head) still works
unchanged. CFDAC inputs (typically 128×128) are bilinearly upsampled to
224×224 — the resolution every backbone was pretrained at.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False


# Friendly backbone name -> timm variant whose ImageNet-1k weights live on a
# reachable GitHub release. Only the top-3 study backbones are wired up.
TIMM_VARIANTS = {
    "resnet50":      "resnet50.a1_in1k",
    "convnext_tiny": "convnext_tiny_hnf.a2h_in1k",
    "vit_b_16":      "vit_base_patch16_224.orig_in21k_ft_in1k",
}

# Kept for backward compatibility with callers that enumerate the sweep.
VISION_BACKBONES = dict(TIMM_VARIANTS)


class VisionBackbone(nn.Module):
    """timm feature-extractor + fresh linear head, with on-the-fly input
    upsampling and a bounded-regression option.

    Interface preserved from the torchvision version so the training /
    eval scripts need no changes:
      * ``self.backbone``  — headless timm model (frozen during the probe
        phase); its parameters are the "backbone" group.
      * ``self.head``      — fresh ``nn.Linear``; always trainable, never
        prefixed with ``backbone.`` so it forms the "head" param group.

    ``channel_adapter`` is accepted for API compatibility but timm handles
    channel adaptation natively (``in_chans``); the value is recorded on
    the artefact and otherwise ignored.
    """

    def __init__(self, name: str, n_channels: int, n_out: int,
                  regression: bool = False, bounded_output: bool = True,
                  pretrained: bool = True, target_size: int = 224,
                  channel_adapter: str = "timm_in_chans"):
        super().__init__()
        if not _HAS_TIMM:
            raise ImportError("timm required for vision backbones")
        if name not in TIMM_VARIANTS:
            raise ValueError(
                f"unknown vision backbone {name!r}; wired variants: "
                f"{list(TIMM_VARIANTS)}")
        variant = TIMM_VARIANTS[name]
        # Force the GitHub-release URL source (null out hf_hub_id) so the
        # blocked HuggingFace hub is never contacted.
        overlay = dict(hf_hub_id=None)
        self.backbone = timm.create_model(
            variant, pretrained=pretrained, in_chans=int(n_channels),
            num_classes=0, pretrained_cfg_overlay=overlay)
        n_feats = int(self.backbone.num_features)
        self.head = nn.Linear(n_feats, n_out)
        self.channel_projector = None  # timm handles channels natively
        self.name = name
        self.variant = variant
        self.regression = regression
        self.bounded_output = bounded_output
        self.target_size = int(target_size)
        self.channel_adapter = channel_adapter
        self.n_features = n_feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"VisionBackbone expects (B, C, H, W), got {x.shape}")
        if x.shape[-1] != self.target_size or x.shape[-2] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode="bilinear",
                                  align_corners=False)
        feats = self.backbone(x)
        out = self.head(feats)
        if self.regression and self.bounded_output:
            out = torch.sigmoid(out)
        return out


def is_vision_backbone(name: str) -> bool:
    return name in TIMM_VARIANTS
