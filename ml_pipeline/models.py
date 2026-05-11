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
                  dropout: float = 0.2, regression: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)
        self.regression = regression

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, F)
        if x.ndim == 3:
            x = x.flatten(1)
        return self.net(x)


# ── 1-D CNN ─────────────────────────────────────────────────────────────────
class Conv1DStack(nn.Module):
    """Stack of Conv1d + BN + GELU + MaxPool, ending in a global-pool head."""

    def __init__(self, n_channels: int, n_out: int,
                  widths: tuple[int, ...] = (32, 64, 128),
                  kernel_size: int = 7, regression: bool = False):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x = self.features(x)
        return self.head(x)


# ── small Transformer ───────────────────────────────────────────────────────
class SmallTransformer(nn.Module):
    """Channel-token Transformer with frequency-axis pooling.

    Input is reshaped to ``(B, L, C)`` then projected to d_model and fed to a
    standard ``nn.TransformerEncoder``.  A CLS token aggregates and the head
    projects to the task output.  Length is downsampled with a strided conv
    so attention stays cheap on 1000-step inputs.
    """

    def __init__(self, n_channels: int, n_out: int,
                  d_model: int = 64, n_heads: int = 4,
                  n_layers: int = 2, downsample: int = 8,
                  regression: bool = False):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        z = self.proj(x).transpose(1, 2)             # (B, L', d_model)
        cls = self.cls.expand(z.size(0), -1, -1)
        z = torch.cat([cls, z], dim=1)
        z = self.encoder(z)
        return self.head(z[:, 0])


def build_torch_model(name: str, in_channels: Optional[int],
                       in_dim: Optional[int], n_out: int,
                       regression: bool = False) -> nn.Module:
    if name == "mlp":
        return MLP(in_dim=in_dim, n_out=n_out, regression=regression)
    if name == "cnn":
        return Conv1DStack(n_channels=in_channels, n_out=n_out, regression=regression)
    if name == "transformer":
        return SmallTransformer(n_channels=in_channels, n_out=n_out,
                                 regression=regression)
    raise ValueError(name)
