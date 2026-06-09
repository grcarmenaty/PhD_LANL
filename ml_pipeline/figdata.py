"""Shared data access for the hi-res FIGURE pipeline — makes every figure
reproducible from data committed in the repo, with graceful fallback to the
raw sources if a developer has them. Resolution-aware: works for the native
1601-bin study and the decimated 128-bin study from the same committed layout.

Committed artefacts (built by `build_figure_bundle.py --res {1601,128}`):
  results_hires/per_case_hires{res}.tar.gz   per-case PREDICTIONS at that res
                                             (GPU outputs; not cheaply regenerable)
  results_hires/figure_data{sfx}.npz         compact cache of the FRF-derived
                                             quantities the EDA/input figures need
                                             (sfx = "" for 1601, "_128" otherwise)

Resolution order is always: committed artefact -> raw source (/tmp/allres or
the hi-res HDF5).
"""
from __future__ import annotations
import tarfile, glob
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
_RH = REPO / "results_hires"
_PC_DIR = _RH / "per_case_all"
_EXP_H5 = REPO / "dataset" / "experimental_features_hires.h5"
_SYN_H5 = REPO / "dataset" / "features_hires.h5"
_LEGACY = Path("/tmp/allres")


def sfx(res: int) -> str:
    """File-name suffix: '' for the native 1601 study, '_{res}' otherwise."""
    return "" if int(res) == 1601 else f"_{int(res)}"


def figdir(res: int) -> Path:
    """results/figures/hires (1601) | results/figures/hires{res} (other)."""
    d = REPO / "results" / "figures" / ("hires" if int(res) == 1601 else f"hires{int(res)}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _bundle(res: int) -> Path:
    return _RH / f"figure_data{sfx(res)}.npz"


# --------------------------------------------------------------------------- per-case predictions
def percase_root() -> str:
    """Directory holding the per-case JSONs (ALL resolutions). Extracts every
    committed per_case_hires*.tar.gz on first use; falls back to /tmp/allres."""
    if _PC_DIR.exists() and any(_PC_DIR.glob("**/*_hires*.json")):
        return str(_PC_DIR)
    tgzs = sorted(glob.glob(str(_RH / "per_case_hires*.tar.gz")))
    if tgzs:
        _PC_DIR.mkdir(parents=True, exist_ok=True)
        for tg in tgzs:
            with tarfile.open(tg, "r:gz") as t:
                t.extractall(_PC_DIR)
        return str(_PC_DIR)
    return str(_LEGACY)


# --------------------------------------------------------------------------- experimental labels (res-independent)
def load_exp_labels():
    """(names, type_code, severity) for the 2 638 exp cases — identical at every
    resolution. Prefers any committed bundle; falls back to the experimental HDF5."""
    for res in (1601, 128):
        b = _bundle(res)
        if b.exists():
            d = np.load(b, allow_pickle=True)
            return [str(s) for s in d["exp_names"]], d["exp_tc"].astype(int), d["exp_sev"].astype(float)
    import h5py
    with h5py.File(_EXP_H5, "r") as f:
        return ([str(s) for s in f["names"][:]], f["type_code"][:].astype(int),
                f["severity"][:].astype(float))


# --------------------------------------------------------------------------- EDA arrays (res-specific)
def load_eda_arrays(res: int = 1601):
    b = _bundle(res)
    if b.exists():
        d = np.load(b, allow_pickle=True)
        return {k: d[k] for k in ("syn_tc", "syn_sev", "syn_lm", "syn_lm_tc", "syn_lm_sev",
                                  "exp_lm", "exp_tc", "exp_sev", "freqs")}
    return None    # caller falls back to computing from HDF5


def load_input_samples(res: int = 1601):
    b = _bundle(res)
    if not b.exists():
        return None
    d = np.load(b, allow_pickle=True)
    out = {}
    for dom in ("synth", "exp"):
        H = {int(k): v for k, v in d[f"{dom}_sampH"].item().items()}
        out[dom] = {"freqs": d["freqs"], "ref": d[f"{dom}_ref"],
                    "idx": d[f"{dom}_idx"].item(), "H": H, "sev": d[f"{dom}_sampsev"].item()}
    return out


def have_bundle(res: int = 1601) -> bool:
    return _bundle(res).exists()
