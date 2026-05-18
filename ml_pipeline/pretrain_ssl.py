"""P2.3: SimCLR-style contrastive pretraining on the 2638 unlabelled
experimental FRF cases.

The current pipeline uses the experimental data only at evaluation
time; the 2638 cases carry a lot of in-domain structural information
about the IQS test rig that no synth augmentation can fully capture.
This script pretrains each backbone (CNN, Conv2D, Conv3D, etc.) with
a contrastive objective so the backbone learns a representation
where two augmentations of the same case are close and two
augmentations of different cases are far apart.  The pretrained
backbone state then warm-starts the synth training in train.py /
hpo.py via the new --init-from flag.

Pipeline:
    python -m ml_pipeline.pretrain_ssl \\
        --exp dataset/experimental_features.h5 \\
        --backbones cnn cnn2d \\
        --features cfdac_realimag frf_mag \\
        --epochs 50 --out results/models_ssl
    python -m ml_pipeline.hpo \\
        --features dataset/features.h5 \\
        --init-from results/models_ssl \\
        --out results_p2_3

Augmentation strategy for FRFs / CFDAC:
  * frequency-band crop: random contiguous sub-band of [70 %, 100 %]
    of full length, then resampled back to original length.
  * channel dropout: zero out 1-2 channels at random.
  * magnitude jitter: per-channel multiplicative scaling in
    U(0.8, 1.25) on a 50 % subset of channels.

The NT-Xent (InfoNCE) loss with temperature 0.1 separates positive
pairs (two views of the same case) from all in-batch negatives.

Output: results/models_ssl/<model>_<feature>_ssl.pt with the
backbone state_dict only (no head -- it gets a fresh head from
hpo.py at fine-tune time).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.models import (
    MLP, Conv1DStack, SmallTransformer, Conv2DStack, Conv3DStack,
)
from ml_pipeline.train import _CFDAC_VARIANTS, _per_sample_normalize


SEED = 20260518


# ── Augmentations ────────────────────────────────────────────────────────────
def aug_freq_crop(x: np.ndarray, rng: np.random.Generator,
                    min_frac: float = 0.70) -> np.ndarray:
    """Random sub-band crop along the leading axis after channel.

    For sequences (n_freq, C) or CFDAC ((C,) H, W), crops the freq /
    rows axis to a random contiguous chunk of fraction f in
    [min_frac, 1.0], then linearly interpolates back to original
    length.  Implemented as bin-averaging the cropped region.
    """
    if x.ndim == 2:                       # (n_freq, C)
        n_freq, C = x.shape
        f = float(rng.uniform(min_frac, 1.0))
        new_n = max(8, int(round(n_freq * f)))
        lo = int(rng.integers(0, n_freq - new_n + 1))
        cropped = x[lo:lo + new_n]
        # zoom back to n_freq via linear interp
        xp = np.arange(new_n)
        x_out = np.empty_like(x)
        for c in range(C):
            x_out[:, c] = np.interp(np.linspace(0, new_n - 1, n_freq),
                                       xp, cropped[:, c])
        return x_out
    if x.ndim == 3:                       # (C, H, W) -- CFDAC
        C, H, W = x.shape
        f = float(rng.uniform(min_frac, 1.0))
        new_h = max(8, int(round(H * f)))
        lo = int(rng.integers(0, H - new_h + 1))
        cropped = x[:, lo:lo + new_h, :]
        # nearest-neighbour zoom back to (H, W)
        idx_h = np.clip(np.linspace(0, new_h - 1, H).round().astype(int),
                          0, new_h - 1)
        return cropped[:, idx_h, :]
    return x


def aug_channel_dropout(x: np.ndarray, rng: np.random.Generator,
                          max_drop: int = 2) -> np.ndarray:
    """Zero out 0-`max_drop` channels at random."""
    if x.ndim == 2:                       # (n_freq, C)
        C = x.shape[1]
        n_drop = int(rng.integers(0, max_drop + 1))
        if n_drop > 0:
            drop_idx = rng.choice(C, size=n_drop, replace=False)
            x = x.copy(); x[:, drop_idx] = 0.0
        return x
    if x.ndim == 3:                       # (C, H, W)
        C = x.shape[0]
        n_drop = int(rng.integers(0, max_drop + 1))
        if n_drop > 0:
            drop_idx = rng.choice(C, size=n_drop, replace=False)
            x = x.copy(); x[drop_idx] = 0.0
        return x
    return x


def aug_magnitude_jitter(x: np.ndarray, rng: np.random.Generator,
                            low: float = 0.8, high: float = 1.25,
                            frac: float = 0.5) -> np.ndarray:
    """Per-channel multiplicative scaling on a `frac` random subset."""
    x = x.copy().astype(np.float32)
    if x.ndim == 2:
        C = x.shape[1]
        mask = rng.uniform(size=C) < frac
        scale = rng.uniform(low, high, size=C).astype(np.float32)
        scale[~mask] = 1.0
        x = x * scale[None, :]
    elif x.ndim == 3:
        C = x.shape[0]
        mask = rng.uniform(size=C) < frac
        scale = rng.uniform(low, high, size=C).astype(np.float32)
        scale[~mask] = 1.0
        x = x * scale[:, None, None]
    return x


def make_two_views(x: np.ndarray, rng: np.random.Generator,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    def _view(x):
        x = aug_freq_crop(x, rng)
        x = aug_channel_dropout(x, rng)
        x = aug_magnitude_jitter(x, rng)
        return x.astype(np.float32)
    return _view(x), _view(x)


# ── Dataset ──────────────────────────────────────────────────────────────────
class _ExpFeatureDataset(Dataset):
    def __init__(self, exp_path: Path, feature: str, seed: int = SEED):
        self.exp_path = exp_path
        self.feature = feature
        with h5py.File(exp_path, "r") as f:
            if feature in _CFDAC_VARIANTS:
                parts, mode = _CFDAC_VARIANTS[feature]
                layers = []
                for p in parts:
                    arr = f[p][:]
                    arr = _per_sample_normalize(p, arr)
                    layers.append(arr)
                self.X = np.stack(layers, axis=1)  # (n, C, H, W)
                if mode == "stack3d":
                    self.X = self.X[:, np.newaxis, ...]
            else:
                self.X = _per_sample_normalize(feature, f[feature][:])
        self.n = len(self.X)
        self.rng = np.random.default_rng(seed)
        # Pre-generate per-worker rngs so the contrastive views are
        # reproducible run-to-run.
        self.worker_rngs = [np.random.default_rng(seed + 1000 + i)
                              for i in range(8)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        # Use a worker-local rng so this is roughly deterministic per
        # epoch+index but still varies across epochs because DataLoader
        # constructs fresh worker rngs each time -- we just want
        # *different* views per call.
        rng = self.worker_rngs[idx % len(self.worker_rngs)]
        v1, v2 = make_two_views(self.X[idx], rng)
        return torch.as_tensor(v1).float(), torch.as_tensor(v2).float()


# ── NT-Xent contrastive loss ─────────────────────────────────────────────────
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                   temperature: float = 0.1) -> torch.Tensor:
    """SimCLR-style InfoNCE on a batch of pairs."""
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)             # (2B, D)
    z = F.normalize(z, dim=1)
    sim = (z @ z.t()) / temperature             # (2B, 2B)
    # mask self-similarity
    eye = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(eye, -1e9)
    # positives: i <-> i+B
    target = torch.cat([torch.arange(B) + B,
                          torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, target)


# ── Backbone with a contrastive projection head ──────────────────────────────
class _BackboneWithProj(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, proj_dim: int = 64):
        super().__init__()
        # Replace the existing head with a projection MLP.
        self.backbone = backbone
        # Pull out the backbone's pre-head feature pipeline.  For each
        # of our model classes the layout is consistent: there's a
        # `self.features` or equivalent and a `self.head` Linear.
        # We just monkey-patch the head to be a projection MLP.
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        # Run the backbone up to (but not through) its final Linear.
        b = self.backbone
        if isinstance(b, MLP):
            if x.ndim == 3: x = x.flatten(1)
            # Strip the last Linear from b.net so we get pre-head features.
            *body, _ = list(b.net.children())
            for m in body: x = m(x)
        elif isinstance(b, Conv1DStack):
            x = b.features(x)
            # Run head up to (but not including) the final Linear.
            *body, _ = list(b.head.children())
            for m in body: x = m(x)
        elif isinstance(b, SmallTransformer):
            z = b.proj(x).transpose(1, 2)
            cls = b.cls.expand(z.size(0), -1, -1)
            z = torch.cat([cls, z], dim=1)
            z = b.encoder(z)
            x = b.head[0](z[:, 0])     # LayerNorm only
        elif isinstance(b, Conv2DStack):
            x = b.stem(x); x = b.features(x)
            *body, _ = list(b.head.children())
            for m in body: x = m(x)
        elif isinstance(b, Conv3DStack):
            x = b.stem(x); x = b.features(x)
            *body, _ = list(b.head.children())
            for m in body: x = m(x)
        else:
            raise ValueError(type(b))
        return self.proj(x)


def _feat_dim(model_name: str) -> int:
    """Width of the backbone's feature vector just before the final Linear."""
    # Matches the constants used in models.py.
    return {"mlp": 64, "cnn": 128, "transformer": 48,
                "cnn2d": 64, "cnn3d": 64}[model_name]


# ── Pretrain driver ──────────────────────────────────────────────────────────
def pretrain_one(model_name: str, feature: str, exp_path: Path,
                    out_dir: Path, epochs: int, batch_size: int,
                    seed: int) -> Path:
    """Pretrain one (model, feature) backbone via SimCLR and save the
    backbone state_dict."""
    ds = _ExpFeatureDataset(exp_path, feature, seed=seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Probe shapes for backbone construction.
    sample, _ = ds[0]
    if model_name == "mlp":
        in_dim = int(np.prod(sample.shape))
        backbone = MLP(in_dim=in_dim, n_out=2,
                          regression=False, bounded_output=True)
    elif model_name == "cnn":
        ch = sample.shape[0] if sample.ndim == 2 else sample.shape[-1]
        backbone = Conv1DStack(n_channels=ch, n_out=2,
                                   regression=False, bounded_output=True)
    elif model_name == "transformer":
        ch = sample.shape[0] if sample.ndim == 2 else sample.shape[-1]
        backbone = SmallTransformer(n_channels=ch, n_out=2,
                                          regression=False,
                                          bounded_output=True)
    elif model_name == "cnn2d":
        backbone = Conv2DStack(n_channels=sample.shape[0], n_out=2,
                                    regression=False, bounded_output=True)
    elif model_name == "cnn3d":
        backbone = Conv3DStack(depth=sample.shape[1], n_out=2,
                                    regression=False, bounded_output=True)
    else:
        raise ValueError(model_name)

    mdl = _BackboneWithProj(backbone, feat_dim=_feat_dim(model_name))
    opt = torch.optim.AdamW(mdl.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"  SSL pretrain {model_name}/{feature}: "
              f"{len(ds)} cases, batch={batch_size}, epochs={epochs}",
              flush=True)
    for ep in range(epochs):
        mdl.train()
        epoch_loss = 0.0
        n_batches = 0
        for v1, v2 in dl:
            opt.zero_grad()
            z1 = mdl(v1); z2 = mdl(v2)
            loss = nt_xent_loss(z1, z2, temperature=0.1)
            loss.backward(); opt.step()
            epoch_loss += float(loss); n_batches += 1
        sched.step()
        if (ep + 1) % max(1, epochs // 10) == 0 or ep == epochs - 1:
            print(f"    epoch {ep + 1}/{epochs}  "
                      f"loss={epoch_loss / max(1, n_batches):.4f}",
                      flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_{feature}_ssl.pt"
    torch.save({"backbone_state_dict": backbone.state_dict(),
                 "model_name": model_name,
                 "feature": feature,
                 "ssl_epochs": epochs,
                 "ssl_temperature": 0.1,
                 "n_cases": len(ds)}, out_path)
    print(f"  wrote {out_path}", flush=True)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", type=Path,
                      default=_REPO / "dataset" / "experimental_features.h5")
    p.add_argument("--out", type=Path,
                      default=_REPO / "results" / "models_ssl")
    p.add_argument("--backbones", nargs="+",
                      default=["cnn", "cnn2d"],
                      choices=("mlp", "cnn", "transformer", "cnn2d", "cnn3d"))
    p.add_argument("--features", nargs="+",
                      default=["frf_mag", "cfdac_realimag"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    print(f"SSL pretraining: {len(args.backbones)} backbones x "
              f"{len(args.features)} features = "
              f"{len(args.backbones) * len(args.features)} cells",
              flush=True)
    for bb in args.backbones:
        for ft in args.features:
            # Skip incompatible combinations.
            is_mat = ft in _CFDAC_VARIANTS and "stack3d" not in str(
                _CFDAC_VARIANTS.get(ft, (None, "")))
            is_3d = ft in _CFDAC_VARIANTS and "stack3d" in str(
                _CFDAC_VARIANTS.get(ft, (None, "")))
            if bb == "cnn2d" and not is_mat:
                print(f"  skip {bb}/{ft}: 2D backbone needs CFDAC feature")
                continue
            if bb == "cnn3d" and not is_3d:
                print(f"  skip {bb}/{ft}: 3D backbone needs cfdac3d_* feature")
                continue
            if bb in ("cnn", "transformer") and (is_mat or is_3d):
                print(f"  skip {bb}/{ft}: 1D backbone needs sequence feature")
                continue
            t0 = time.time()
            pretrain_one(bb, ft, args.exp, args.out, args.epochs,
                            args.batch_size, args.seed)
            print(f"  {bb}/{ft} done in {time.time() - t0:.1f}s",
                      flush=True)


if __name__ == "__main__":
    main()
