"""Lazy HDF5-backed datasets so feature arrays don't have to fit in RAM.

The flat ``dataset/features.h5`` produced by ``ml_pipeline.features``
keeps each CFDAC variant on disk as a gzip-compressed chunked dataset
of shape ``(n, H, W)``.  Loading the whole 10 000-row array eagerly
costs 0.7 – 2.6 GB of RAM and 1 – 3 min of gzip decompression per
variant; for a grid-HPO sweep that pre-load cost dominates the
wall-clock.

The classes below wrap one or more HDF5 datasets behind the standard
``torch.utils.data.Dataset`` interface so the trainer can use a
DataLoader and only the current mini-batch ever lives in RAM.

Example
-------
>>> from ml_pipeline.lazy_datasets import LazyCFDACDataset
>>> ds = LazyCFDACDataset("dataset/features.h5", "cfdac_realimag",
...                          rows=train_idx, labels=y_train, kind="cls")
>>> dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True,
...                                       num_workers=2, persistent_workers=True)
>>> for xb, yb in dl: ...
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Same catalogue used in train.py — duplicated rather than imported to
# avoid the cyclic dependency through ``train``.
_CFDAC_PARTS = {
    "cfdac_real":     (("cfdac_real",),                          "stack2d"),
    "cfdac_imag":     (("cfdac_imag",),                          "stack2d"),
    "cfdac_mag":      (("cfdac_mag",),                           "stack2d"),
    "cfdac_phase":    (("cfdac_phase",),                         "stack2d"),
    "cfdac_realimag": (("cfdac_real", "cfdac_imag"),             "stack2d"),
    "cfdac_magphase": (("cfdac_mag",  "cfdac_phase"),            "stack2d"),
    "cfdac_all":      (("cfdac_real", "cfdac_imag",
                          "cfdac_mag",  "cfdac_phase"),            "stack2d"),
    "cfdac3d_realimag": (("cfdac_real", "cfdac_imag"),                  "stack3d"),
    "cfdac3d_magphase": (("cfdac_mag",  "cfdac_phase"),                 "stack3d"),
    "cfdac3d_all":      (("cfdac_real", "cfdac_imag",
                            "cfdac_mag",  "cfdac_phase"),                "stack3d"),
}


class LazyCFDACDataset(Dataset):
    """Streams one CFDAC variant from disk row-by-row.

    Each ``__getitem__`` opens its own file handle (so workers can fork
    safely under ``num_workers > 0``), reads the requested row from
    every part of the variant, stacks them into the canonical
    Conv2d / Conv3d layout, and returns ``(x, y)`` tensors.

    Parameters
    ----------
    h5_path : Path
        ``features.h5`` (or ``experimental_features.h5``) produced by
        the pipeline.
    variant : str
        Key into ``_CFDAC_PARTS`` — one of the 10 CFDAC variants.
    rows : 1-D numpy array, optional
        Subset of row indices to expose; if None, the dataset spans
        every row in the file.
    labels : 1-D numpy array, optional
        ``y`` values aligned with ``rows``.  For classification this
        is int64; for regression float32.  If None, the dataset only
        returns ``x`` tensors.
    kind : str
        "cls" → labels are returned as long tensors; "reg" → floats.
    """

    def __init__(self, h5_path, variant: str,
                  rows: np.ndarray | None = None,
                  labels: np.ndarray | None = None,
                  kind: str = "cls"):
        if variant not in _CFDAC_PARTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.h5_path = str(h5_path)
        self.parts, self.mode = _CFDAC_PARTS[variant]
        # Probe shape without loading.
        with h5py.File(self.h5_path, "r") as f:
            n_total = f[self.parts[0]].shape[0]
            self.h, self.w = f[self.parts[0]].shape[1:]
        self.rows = np.asarray(rows) if rows is not None else np.arange(n_total)
        self.labels = labels
        self.kind = kind

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = int(self.rows[i])
        with h5py.File(self.h5_path, "r") as f:
            layers = [f[p][row] for p in self.parts]
        x = np.stack(layers, axis=0).astype(np.float32)   # (C, H, W)
        if self.mode == "stack3d":
            x = x[np.newaxis, ...]                          # (1, D, H, W)
        if self.labels is None:
            return torch.from_numpy(x)
        y = self.labels[i]
        if self.kind == "cls":
            y_t = torch.tensor(int(y), dtype=torch.long)
        else:
            y_t = torch.tensor(float(y), dtype=torch.float32).unsqueeze(0)
        return torch.from_numpy(x), y_t

    def batch_read(self, rows: Sequence[int]) -> np.ndarray:
        """Bulk-read a set of rows; used by the trainer to construct the
        validation / test tensors without spawning DataLoader workers."""
        rows = np.asarray(rows, dtype=int)
        order = np.argsort(rows)
        with h5py.File(self.h5_path, "r") as f:
            layers = []
            for p in self.parts:
                tmp = f[p][rows[order]]
                arr = np.empty_like(tmp); arr[order] = tmp
                layers.append(arr)
        x = np.stack(layers, axis=1).astype(np.float32)   # (n, C, H, W)
        if self.mode == "stack3d":
            x = x[:, np.newaxis, ...]                       # (n, 1, D, H, W)
        return x


__all__ = ["LazyCFDACDataset"]
