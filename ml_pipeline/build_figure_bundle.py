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
from ml_pipeline.figdata import sfx
SYN = REPO / "dataset" / "features_hires.h5"
EXP = REPO / "dataset" / "experimental_features_hires.h5"


def build_percase_archive(root, res):
    """Dedupe the `res`-bin per-case JSONs (one per task/model/feature) into a
    flat gzip tarball. Keeps the first occurrence of each filename."""
    out = REPO / "results_hires" / f"per_case_hires{res}.tar.gz"
    seen = {}
    for p in glob.glob(f"{root}/**/per_case/*_hires{res}.json", recursive=True):
        name = Path(p).name
        if name not in seen:
            seen[name] = p
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as t:
        for name, p in sorted(seen.items()):
            t.add(p, arcname=f"per_case/{name}")
    print(f"per-case archive: {len(seen)} unique files -> {out.name} ({out.stat().st_size/1048576:.1f} MB)")


def build_data_bundle(res):
    """Compact cache of everything the EDA + input-sample figures read from the
    1.2 GB hi-res HDF5 (so they regenerate without it), at frequency res `res`."""
    import h5py
    out = REPO / "results_hires" / f"figure_data{sfx(res)}.npz"
    syn_lm, syn_lm_tc, syn_lm_sev, freqs = logmag_chanmean(SYN, sub=4000, res=res)
    exp_lm, exp_tc, exp_sev, _ = logmag_chanmean(EXP, sub=None, res=res)
    with h5py.File(SYN, "r") as f:
        syn_tc = f["type_code"][:].astype(np.int16); syn_sev = f["severity"][:].astype(np.float32)
    S = pick(SYN, res=res); E = pick(EXP, res=res)
    def packH(D):
        return {int(c): D["H"][c] for c in D["H"]}
    np.savez_compressed(
        out,
        freqs=freqs.astype(np.float32),
        syn_tc=syn_tc, syn_sev=syn_sev,                                    # full (10 000)
        syn_lm=syn_lm.astype(np.float32), syn_lm_tc=syn_lm_tc.astype(np.int16),
        syn_lm_sev=syn_lm_sev.astype(np.float32),                          # subsample (4 000)
        exp_lm=exp_lm.astype(np.float32), exp_tc=exp_tc.astype(np.int16), exp_sev=exp_sev.astype(np.float32),
        exp_names=np.array(_names(EXP), dtype=object),
        synth_ref=S["ref"], synth_idx=S["idx"], synth_sampH=packH(S), synth_sampsev=S["sev"],
        exp_ref=E["ref"], exp_idx=E["idx"], exp_sampH=packH(E), exp_sampsev=E["sev"],
    )
    print(f"figure data bundle -> {out.name} ({out.stat().st_size/1048576:.1f} MB)  "
          f"[syn_lm {syn_lm.shape}, exp_lm {exp_lm.shape}]")


def _names(h5):
    import h5py
    with h5py.File(h5, "r") as f:
        return [str(s) for s in f["names"][:]] if "names" in f else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/allres")
    ap.add_argument("--res", type=int, default=1601)
    a = ap.parse_args()
    build_percase_archive(a.root, a.res)
    build_data_bundle(a.res)
    print(f"done — commit per_case_hires{a.res}.tar.gz + figure_data{sfx(a.res)}.npz")


if __name__ == "__main__":
    main()
