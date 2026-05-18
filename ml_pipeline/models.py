"""Model definitions for the 4 architectures used in this study.

Random Forest, XGBoost and MLP are trained via scikit-learn / xgboost
(flat feature inputs).  1-D CNN and Transformer are PyTorch modules
operating on shape ``(batch, channels, length)``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn


# ── MLP ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int, n_out: int,
                  hidden: tuple[int, ...] = (256, 128, 64),
                  dropout: float = 0.2, regression: bool = False,
                  bounded_output: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)
        self.regression = regression
        self.bounded_output = bounded_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, F)
        if x.ndim == 3:
            x = x.flatten(1)
        out = self.net(x)
        return torch.sigmoid(out) if (self.regression and self.bounded_output) else out


# ── 1-D CNN ─────────────────────────────────────────────────────────────────
class Conv1DStack(nn.Module):
    """Stack of Conv1d + BN + GELU + MaxPool, ending in a global-pool head."""

    def __init__(self, n_channels: int, n_out: int,
                  widths: tuple[int, ...] = (32, 64, 128),
                  kernel_size: int = 7, regression: bool = False,
                  bounded_output: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        c = n_channels
        for w in widths:
            layers += [
                nn.Conv1d(c, w, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(w),
                nn.GELU(),
                nn.MaxPool1d(2),
            ]
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_out),
        )
        self.regression = regression
        self.bounded_output = bounded_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x = self.features(x)
        out = self.head(x)
        return torch.sigmoid(out) if (self.regression and self.bounded_output) else out


# ── small Transformer ───────────────────────────────────────────────────────
class SmallTransformer(nn.Module):
    """Channel-token Transformer with frequency-axis pooling.

    Input is reshaped to ``(B, L, C)`` then projected to d_model and fed to a
    standard ``nn.TransformerEncoder``.  A CLS token aggregates and the head
    projects to the task output.  Length is downsampled with a strided conv
    so attention stays cheap on 1000-step inputs.
    """

    def __init__(self, n_channels: int, n_out: int,
                  d_model: int = 48, n_heads: int = 4,
                  n_layers: int = 2, downsample: int = 16,
                  regression: bool = False,
                  bounded_output: bool = True):
        super().__init__()
        self.downsample = downsample
        self.proj = nn.Conv1d(n_channels, d_model, kernel_size=downsample,
                                stride=downsample)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, n_out),
        )
        self.regression = regression
        self.bounded_output = bounded_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        z = self.proj(x).transpose(1, 2)             # (B, L', d_model)
        cls = self.cls.expand(z.size(0), -1, -1)
        z = torch.cat([cls, z], dim=1)
        z = self.encoder(z)
        out = self.head(z[:, 0])
        return torch.sigmoid(out) if (self.regression and self.bounded_output) else out


class Conv2DStack(nn.Module):
    """2-D CNN for matricial features (CFDAC).

    Input shape: ``(B, 2, 128, 128)`` (real + imag CFDAC).  Architecture:

      * one strided stem (kernel = 7, stride = 4) that drops the
        spatial extent to 32×32 immediately — required to keep CPU
        training time tractable;
      * three Conv2d + BN + GELU + MaxPool2d blocks that shrink to
        16×16 → 8×8 → 4×4 with channel widths ``widths``;
      * global average pool + two-layer MLP head.
    """

    def __init__(self, n_channels: int, n_out: int,
                  widths: tuple[int, ...] = (16, 32, 64),
                  kernel_size: int = 5,
                  regression: bool = False,
                  bounded_output: bool = True):
        super().__init__()
        c = n_channels
        stem_out = widths[0]
        self.stem = nn.Sequential(
            nn.Conv2d(c, stem_out, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(stem_out),
            nn.GELU(),
        )
        c = stem_out
        layers: list[nn.Module] = []
        for w in widths:
            layers += [
                nn.Conv2d(c, w, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(w),
                nn.GELU(),
                nn.MaxPool2d(2),
            ]
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_out),
        )
        self.regression = regression
        self.bounded_output = bounded_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W).  Real/imag CFDAC sit in [-1, 1]; mag in [0, 1];
        # phase in [-pi, pi].  Per-sample normalisation lands in P1.1.
        x = self.stem(x)
        x = self.features(x)
        out = self.head(x)
        return torch.sigmoid(out) if (self.regression and self.bounded_output) else out


class Conv3DStack(nn.Module):
    """3-D CNN for CFDAC parts stacked along an extra "depth" axis.

    Input shape: ``(B, 1, D, H, W)`` where D is the number of CFDAC
    parts (real, imag, mag, phase, …) treated as the depth axis of a
    volumetric input.  The stem is a strided 3-D convolution
    (kernel = (D, 7, 7), stride = (1, 4, 4)) that immediately reduces
    the spatial extent to 32 × 32 while preserving depth.  Subsequent
    blocks pool the spatial dims with MaxPool3d kernel = (1, 2, 2).
    """

    def __init__(self, depth: int, n_out: int,
                  widths: tuple[int, ...] = (8, 16, 32),
                  kernel_size: int = 3, regression: bool = False,
                  bounded_output: bool = True):
        super().__init__()
        self.depth = depth
        stem_out = widths[0]
        self.stem = nn.Sequential(
            nn.Conv3d(1, stem_out, kernel_size=(depth, 7, 7),
                         stride=(1, 4, 4), padding=(0, 3, 3)),
            nn.BatchNorm3d(stem_out),
            nn.GELU(),
        )
        c = stem_out
        layers: list[nn.Module] = []
        for w in widths:
            layers += [
                nn.Conv3d(c, w, kernel_size=(1, kernel_size, kernel_size),
                            padding=(0, kernel_size // 2, kernel_size // 2)),
                nn.BatchNorm3d(w),
                nn.GELU(),
                nn.MaxPool3d(kernel_size=(1, 2, 2)),
            ]
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(c, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_out),
        )
        self.regression = regression
        self.bounded_output = bounded_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, D, H, W)
        x = self.stem(x)
        x = self.features(x)
        out = self.head(x)
        return torch.sigmoid(out) if (self.regression and self.bounded_output) else out


def build_torch_model(name: str, in_channels: Optional[int],
                       in_dim: Optional[int], n_out: int,
                       regression: bool = False,
                       **kwargs) -> nn.Module:
    if name == "mlp":
        return MLP(in_dim=in_dim, n_out=n_out, regression=regression, **kwargs)
    if name == "cnn":
        return Conv1DStack(n_channels=in_channels, n_out=n_out,
                              regression=regression, **kwargs)
    if name == "transformer":
        return SmallTransformer(n_channels=in_channels, n_out=n_out,
                                   regression=regression, **kwargs)
    if name == "cnn2d":
        return Conv2DStack(n_channels=in_channels, n_out=n_out,
                              regression=regression, **kwargs)
    if name == "cnn3d":
        # in_channels here is the depth of the volume, not channel count.
        return Conv3DStack(depth=in_channels, n_out=n_out,
                              regression=regression, **kwargs)
    raise ValueError(name)
