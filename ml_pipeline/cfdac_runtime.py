"""On-the-fly CFDAC at any resolution from FRFs already on disk.

The disk-resident CFDAC (built by `cfdac.py`) is decimated to 128². For
the higher-resolution preliminary study we recompute CFDAC straight from
the FRFs at the requested target_n (up to 381 — the native FRF length).
This keeps disk usage flat: a 381² × 2-channel × 10 000-sample tensor is
~11 GB, way too large for the current container, but per (variant, seed)
we only need ~1 500 samples (the synth subsample) plus the 2 638-case exp
set — well under 5 GB combined.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO.parent / "pymodal") not in sys.path:
    sys.path.insert(0, str(_REPO.parent / "pymodal"))

from pymodal import utils as pm_utils


def _decimate(H: np.ndarray, n_target: int) -> np.ndarray:
    """Bin-average ``H`` from (n_freq, ...) to (n_target, ...)."""
    n_freq = H.shape[0]
    if n_target == n_freq:
        return H
    edges = np.linspace(0, n_freq, n_target + 1).astype(int)
    out = np.empty((n_target,) + H.shape[1:], dtype=H.dtype)
    for i in range(n_target):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        out[i] = H[lo:hi].mean(axis=0)
    return out


def compute_cfdac_runtime(features_path: Path,
                           rows: np.ndarray | None = None,
                           n_target: int = 381,
                           channels: Iterable[str] = ("real", "imag"),
                           ) -> np.ndarray:
    """Compute CFDAC at ``n_target`` resolution for the rows of
    ``features_path``.

    Returns: ``(n, len(channels), n_target, n_target)`` float32.

    ``channels`` is a subset of {"real", "imag", "mag", "phase"}.
    """
    channels = tuple(channels)
    valid = {"real", "imag", "mag", "phase"}
    bad = set(channels) - valid
    if bad:
        raise ValueError(f"unknown CFDAC channels: {bad}")

    with h5py.File(features_path, "r") as f:
        Href_full = f["reference/frf_complex"][:]   # (N_F, 9) complex
        n_disk = f["frf_real"].shape[0]
        if rows is None:
            rows = np.arange(n_disk)
        rows = np.asarray(rows, dtype=np.int64)
        # h5py needs sorted unique indices for fancy indexing; preserve original
        order = np.argsort(rows)
        sorted_rows = rows[order]
        re = f["frf_real"][sorted_rows]              # (k, N_F, 9)
        im = f["frf_imag"][sorted_rows]

    H_ref_d = _decimate(Href_full.astype(np.complex64), n_target)  # (N, 9)

    k = len(rows)
    out = np.empty((k, len(channels), n_target, n_target), dtype=np.float32)
    H_complex = (re + 1j * im).astype(np.complex64)               # (k, N_F, 9)
    for i_sorted in range(k):
        H_d = _decimate(H_complex[i_sorted], n_target)            # (N, 9)
        c = pm_utils.value_CFDAC(H_ref_d, H_d)                    # (N, N) complex
        i_dst = order[i_sorted]   # write back into original row order
        for j, ch in enumerate(channels):
            if ch == "real":  out[i_dst, j] = c.real
            elif ch == "imag": out[i_dst, j] = c.imag
            elif ch == "mag":  out[i_dst, j] = np.abs(c)
            elif ch == "phase": out[i_dst, j] = np.angle(c)
    return out


def per_sample_normalize_cfdac(X: np.ndarray,
                                channels: Iterable[str]) -> np.ndarray:
    """Mirror the per-sample normalisation in ml_pipeline.train so synth-
    train and exp-eval cross-domain inputs share statistics. Operates on
    the (n, C, H, W) tensor returned by :func:`compute_cfdac_runtime`."""
    channels = tuple(channels)
    X = X.astype(np.float32, copy=True)
    for j, ch in enumerate(channels):
        if ch in ("real", "imag"):
            mu = X[:, j].mean(axis=(-2, -1), keepdims=True)
            X[:, j] = X[:, j] - mu[:, 0]
        elif ch == "mag":
            xs = 2.0 * X[:, j] - 1.0
            mu = xs.mean(axis=(-2, -1), keepdims=True)
            X[:, j] = xs - mu[:, 0]
        elif ch == "phase":
            X[:, j] = X[:, j] / np.float32(np.pi)
    return X
