"""Unified runner for the whole hi-res study in ONE grid: every feature family
(non-image -> vision) × every model × every resolution × every task.

Dispatches each cell to the right engine:
  * tabular / non-image models (mlp, rf, xgb, cnn1d, transformer1d) on
    modal / indicators / frf_mag / frf_realimag / timeseries  -> hires_tab
  * CFDAC-image models (cnn2d_shallow/deep, cnn3d, transformer, vision
    backbones) on the 7 CFDAC channel-features                 -> hires_zoo

Both engines take a resolution (`res` / `res_bins`) that decimates the 1601-bin
features, so the SAME cells exist at full and reduced resolution -> the clean
"does full resolution help?" comparison. Per-case tags embed the resolution
(`..._hires{res}`) so nothing collides.
"""
from __future__ import annotations
from typing import List, Sequence, Tuple

from ml_pipeline import hires_zoo as Z
from ml_pipeline import hires_tab as T

IMG_MODELS = ("cnn2d_shallow", "cnn2d_deep", "cnn3d", "transformer",
              "convnext_tiny", "resnet50")
TAB_MODELS = ("mlp", "rf", "xgb", "cnn1d", "transformer1d")


def all_cells(tasks: Sequence[str], resolutions: Sequence[int],
              img_models: Sequence[str] = IMG_MODELS,
              tab_models: Sequence[str] = TAB_MODELS,
              cfdac_features: Sequence[str] | None = None
              ) -> List[Tuple[str, str, str, int]]:
    """Full (task, model, feature, res) grid. Returns tab cells then img cells
    (cheap-first), per resolution."""
    cf = list(Z.CFDAC_FEATURES) if cfdac_features is None else list(cfdac_features)
    out = []
    for res in resolutions:
        for t in tasks:
            for m in tab_models:
                for f in T.TAB_MODEL_FEATURES.get(m, []):
                    out.append((t, m, f, res))
            for m in img_models:
                for f in cf:
                    out.append((t, m, f, res))
    return out


def features_needed(cells) -> set:
    """Non-image (tab) features that need a precomputed cache, with their res."""
    return {(f, res) for (t, m, f, res) in cells if m in TAB_MODELS}


def run_one(task, model, feature, res, *, ctx, log=print):
    """ctx is a dict built once in the notebook (see hires_all_gpu.ipynb)."""
    if model in TAB_MODELS:
        Xs, Xe = ctx["tab_cache"][(feature, res)]
        T.run_tab_cell(task, model, feature, out_dir=ctx["out"], dev=ctx["dev"],
                       syn_tasks=ctx["syn_tasks"], exp_tasks=ctx["exp_tasks"],
                       Xsyn=Xs, Xexp=Xe, exp_names=ctx["exp_names"],
                       make_split=ctx["make_split"], subsample=ctx["subsample"],
                       batch=ctx["tab_batch"], res=res, log=log)
    else:
        Z.run_cell(task, model, feature, syn_h5=ctx["syn_h5"], exp_h5=ctx["exp_h5"],
                   out_dir=ctx["out"], dev=ctx["dev"], syn_tasks=ctx["syn_tasks"],
                   exp_tasks=ctx["exp_tasks"], H_ref_syn=ctx["H_ref_syn"],
                   H_ref_exp=ctx["H_ref_exp"], H_exp=ctx["H_exp"], exp_names=ctx["exp_names"],
                   make_split=ctx["make_split"], subsample=ctx["subsample"],
                   batch=ctx["img_batch"], vision_size=ctx["vision_size"], res_bins=res,
                   log=log)
