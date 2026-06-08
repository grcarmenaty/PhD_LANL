"""Shared data access for the hi-res FIGURE pipeline — makes every figure
reproducible from data committed in the repo, with graceful fallback to the
raw sources if a developer has them.

Two committed artefacts (built by `build_figure_bundle.py`):
  results_hires/per_case_hires1601.tar.gz   the 1601-bin per-case PREDICTIONS
                                            (GPU training outputs; the only data
                                            not cheaply regenerable on CPU)
  results_hires/figure_data.npz             a compact cache of exactly the
                                            FRF-derived quantities the EDA /
                                            input-sample figures need (so the
                                            1.2 GB hi-res HDF5 files do NOT have
                                            to be regenerated just to plot)

Resolution order is always: committed artefact -> raw source (/tmp/allres or
the hi-res HDF5). So the scripts work both on a fresh clone and in the original
GPU/extraction environment.
"""
from __future__ import annotations
import tarfile
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
_PC_DIR = REPO / "results_hires" / "per_case_all"
_PC_TGZ = REPO / "results_hires" / "per_case_hires1601.tar.gz"
_BUNDLE = REPO / "results_hires" / "figure_data.npz"
_EXP_H5 = REPO / "dataset" / "experimental_features_hires.h5"
_SYN_H5 = REPO / "dataset" / "features_hires.h5"
_LEGACY = Path("/tmp/allres")


# --------------------------------------------------------------------------- per-case predictions
def percase_root() -> str:
    """Directory holding the 1601-bin per-case JSONs. Extracts the committed
    archive on first use; falls back to /tmp/allres for the original env."""
    if _PC_DIR.exists() and any(_PC_DIR.glob("**/*_hires1601.json")):
        return str(_PC_DIR)
    if _PC_TGZ.exists():
        _PC_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(_PC_TGZ, "r:gz") as t:
            t.extractall(_PC_DIR)
        return str(_PC_DIR)
    return str(_LEGACY)


# --------------------------------------------------------------------------- experimental labels
def load_exp_labels():
    """(names:list[str], type_code:int[], severity:float[]) for the 2 638 exp
    cases. Prefers the committed bundle; falls back to the experimental HDF5."""
    if _BUNDLE.exists():
        d = np.load(_BUNDLE, allow_pickle=True)
        return [str(s) for s in d["exp_names"]], d["exp_tc"].astype(int), d["exp_sev"].astype(float)
    import h5py
    with h5py.File(_EXP_H5, "r") as f:
        return ([str(s) for s in f["names"][:]], f["type_code"][:].astype(int),
                f["severity"][:].astype(float))


# --------------------------------------------------------------------------- EDA arrays
def load_eda_arrays():
    """Channel-averaged log10|FRF| arrays + labels for the EDA figures.
    Keys: full synth labels syn_tc/syn_sev (10 000); synth subsample for the
    FRF-mean/PCA syn_lm/syn_lm_tc/syn_lm_sev (4 000); full exp exp_lm/exp_tc/
    exp_sev (2 638); freqs. Prefers the committed bundle; None -> compute from HDF5."""
    if _BUNDLE.exists():
        d = np.load(_BUNDLE, allow_pickle=True)
        return {k: d[k] for k in ("syn_tc", "syn_sev", "syn_lm", "syn_lm_tc", "syn_lm_sev",
                                  "exp_lm", "exp_tc", "exp_sev", "freqs")}
    return None    # caller falls back to computing from HDF5


def load_input_samples():
    """dict for the input-sample figures: per domain {freqs, ref(N,9)c, idx{},
    H{c:(N,9)c}, sev{}}. Prefers the bundle; None -> caller reads HDF5."""
    if not _BUNDLE.exists():
        return None
    d = np.load(_BUNDLE, allow_pickle=True)
    out = {}
    for dom in ("synth", "exp"):
        H = {int(k): v for k, v in d[f"{dom}_sampH"].item().items()}
        out[dom] = {"freqs": d["freqs"], "ref": d[f"{dom}_ref"],
                    "idx": d[f"{dom}_idx"].item(), "H": H, "sev": d[f"{dom}_sampsev"].item()}
    return out


def have_bundle() -> bool:
    return _BUNDLE.exists()
