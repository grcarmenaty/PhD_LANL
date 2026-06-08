"""Build the committed FIGURE-reproducibility artefacts from the raw sources.

Run ONCE in the environment that has the GPU per-case outputs (/tmp/allres) and
the hi-res HDF5 files. It writes two committed files that let every figure be
regenerated from the repo alone:

  results_hires/per_case_hires1601.tar.gz   deduped 1601-bin per-case JSONs
  results_hires/figure_data.npz             compact FRF-derived arrays + samples

Run: python ml_pipeline/build_figure_bundle.py --root /tmp/allres
"""
from __future__ import annotations
import argparse, glob, tarfile, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ml_pipeline.hires_analysis import logmag_chanmean      # identical reduction the EDA uses
from ml_pipeline.hires_inputs import pick                   # identical sample selection
SYN = REPO / "dataset" / "features_hires.h5"
EXP = REPO / "dataset" / "experimental_features_hires.h5"
OUT_TGZ = REPO / "results_hires" / "per_case_hires1601.tar.gz"
OUT_NPZ = REPO / "results_hires" / "figure_data.npz"


def build_percase_archive(root):
    """Dedupe the 1601-bin per-case JSONs (one per task/model/feature) into a
    flat gzip tarball. Keeps the first occurrence of each filename."""
    seen = {}
    for p in glob.glob(f"{root}/**/per_case/*_hires1601.json", recursive=True):
        name = Path(p).name
        if name not in seen:
            seen[name] = p
    OUT_TGZ.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OUT_TGZ, "w:gz") as t:
        for name, p in sorted(seen.items()):
            t.add(p, arcname=f"per_case/{name}")
    mb = OUT_TGZ.stat().st_size / 1048576
    print(f"per-case archive: {len(seen)} unique files -> {OUT_TGZ.name} ({mb:.1f} MB)")


def build_data_bundle():
    """Compact cache of everything the EDA + input-sample figures read from the
    1.2 GB hi-res HDF5 (so they regenerate without it)."""
    import h5py
    # FRF-mean reductions (identical seed/subsample to hires_analysis.eda):
    # synth uses a 4 000 subsample for signatures/PCA; exp uses all 2 638.
    syn_lm, syn_lm_tc, syn_lm_sev, freqs = logmag_chanmean(SYN, sub=4000)
    exp_lm, exp_tc, exp_sev, _ = logmag_chanmean(EXP, sub=None)
    # full synth labels (10 000) for the class-balance + severity-distribution panels
    with h5py.File(SYN, "r") as f:
        syn_tc = f["type_code"][:].astype(np.int16); syn_sev = f["severity"][:].astype(np.float32)
    # input-sample selections (identical to hires_inputs.pick)
    S = pick(SYN); E = pick(EXP)
    def packH(D):
        return {int(c): D["H"][c] for c in D["H"]}
    np.savez_compressed(
        OUT_NPZ,
        freqs=freqs.astype(np.float32),
        syn_tc=syn_tc, syn_sev=syn_sev,                                    # full (10 000)
        syn_lm=syn_lm.astype(np.float32), syn_lm_tc=syn_lm_tc.astype(np.int16),
        syn_lm_sev=syn_lm_sev.astype(np.float32),                          # subsample (4 000)
        exp_lm=exp_lm.astype(np.float32), exp_tc=exp_tc.astype(np.int16), exp_sev=exp_sev.astype(np.float32),
        exp_names=np.array(_names(EXP), dtype=object),
        synth_ref=S["ref"], synth_idx=S["idx"], synth_sampH=packH(S), synth_sampsev=S["sev"],
        exp_ref=E["ref"], exp_idx=E["idx"], exp_sampH=packH(E), exp_sampsev=E["sev"],
    )
    mb = OUT_NPZ.stat().st_size / 1048576
    print(f"figure data bundle -> {OUT_NPZ.name} ({mb:.1f} MB)  "
          f"[syn_lm {syn_lm.shape}, exp_lm {exp_lm.shape}]")


def _names(h5):
    import h5py
    with h5py.File(h5, "r") as f:
        return [str(s) for s in f["names"][:]] if "names" in f else []


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/tmp/allres"); a = ap.parse_args()
    build_percase_archive(a.root)
    build_data_bundle()
    print("done — commit results_hires/per_case_hires1601.tar.gz + results_hires/figure_data.npz")


if __name__ == "__main__":
    main()
