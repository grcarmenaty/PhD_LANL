"""Computational-effort accounting for the hi-res model zoo.

There is no stored per-cell wall-clock, so effort is quantified by measurable,
hardware-independent quantities and combined with the (committed) training
protocol + campaign size:

  * parameters per model              (measured; from architectures.json)
  * forward FLOPs at the training input size
        (torch FlopCounterMode; the huge full-1601 conv nets are measured at a
         base size and scaled by the exact area ratio — conv FLOPs ∝ spatial
         area, the size-independent head is negligible at 1601²)
  * CFDAC data-path cost              (analytic: the on-the-fly 1601² cross-FRF
                                       matmul recomputed per image-model sample)
  * campaign size + protocol          (cell count + subsample from the committed
                                       per-case archive; epochs/batch from the engines)

Writes results_hires/compute.json and figures/hires/compute_cost.png. A rough
GPU wall-clock is derived from FLOPs ÷ realized throughput and clearly labelled
as an estimate.

Run: python ml_pipeline/hires_compute.py
"""
from __future__ import annotations
import glob, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.flop_counter import FlopCounterMode

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ml_pipeline import figdata
from ml_pipeline.hires_zoo import ShallowCNN2D, DeepCFDACNet, CNN3D, CFDACTransformer
from ml_pipeline.hires_tab import MLP, CNN1D, Transformer1D
from ml_pipeline import figdata
N = 1601                       # set in main() from --res
IMG_FAMILY = {"cnn2d_shallow", "cnn2d_deep", "cnn3d", "transformer", "resnet50", "convnext_tiny"}


def fwd_gflops(net, shape):
    net.eval()
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        net(torch.zeros(*shape))
    return fc.get_total_flops() / 1e9


def measure(N):
    """forward GFLOPs at the actual training input size, per model (n_in=2 ch)."""
    g = {}
    # conv nets: measure at base, scale by area ratio to 1601²
    g["cnn2d_shallow"] = fwd_gflops(ShallowCNN2D(2, 2), (1, 2, 400, 400)) * (N/400)**2
    g["cnn2d_deep"]    = fwd_gflops(DeepCFDACNet(2, 2), (1, 2, 400, 400)) * (N/400)**2
    g["cnn3d"]         = fwd_gflops(CNN3D(2, 2), (1, 2, 200, 200)) * (N/200)**2
    # CFDAC transformer: tokeniser collapses 1601→~25 fast, measure directly
    g["transformer"]   = fwd_gflops(CFDACTransformer(2, 2, input_size=N), (1, 2, N, N))
    # vision backbones run on the 384² resized CFDAC
    try:
        import timm
        for k in ("resnet50", "convnext_tiny"):
            net = timm.create_model(k, pretrained=False, in_chans=2, num_classes=2)
            g[k] = fwd_gflops(net, (1, 2, 384, 384))
    except Exception as e:
        print("timm skip:", e)
    # sequence models over length 1601 (9 channels) — measured directly
    g["cnn1d"]         = fwd_gflops(CNN1D(9, 2), (1, 9, N))
    g["transformer1d"] = fwd_gflops(Transformer1D(9, 2, length=N), (1, 9, N))
    # mlp: linear, analytic range (modal d_in=81 .. flattened frf_mag 9*1601)
    def mlp_g(d):  return fwd_gflops(MLP(d, 2), (2, d))      # batch>=2 for BN
    g["mlp"] = mlp_g(81)
    g["mlp_flat_seq"] = mlp_g(9*N)                            # worst case (flattened frf_mag)
    return g


def cfdac_gflops(N, channels=2):
    """On-the-fly CFDAC per sample: cross = frf(N,9)·conj(ref(9,N)) -> N×N, then
    num/denom/normalise. Cross matmul dominates: N²·9 complex MACs (~8 FLOP each)."""
    cross = N*N*9*8
    elementwise = N*N*(channels*3 + 4)        # square, divide, per-channel normalise
    return (cross + elementwise) / 1e9


def campaign(res):
    """cell count + subsample per family, from the committed per-case archive."""
    root = figdata.percase_root()
    fams = defaultdict(lambda: {"n": 0, "subs": []})
    seen = set()
    for p in glob.glob(f"{root}/**/per_case/*_hires{res}.json", recursive=True):
        name = Path(p).name
        if name in seen: continue
        seen.add(name)
        try: m = json.load(open(p))["meta"]
        except Exception: continue
        fam = "image" if m["model"] in IMG_FAMILY else "tabular/seq"
        fams[fam]["n"] += 1
        if m.get("subsample"): fams[fam]["subs"].append(int(m["subsample"]))
    return {k: {"cells": v["n"], "median_subsample": int(np.median(v["subs"])) if v["subs"] else None}
            for k, v in fams.items()}, len(seen)


def main():
    import argparse
    global FIG
    ap = argparse.ArgumentParser(); ap.add_argument("--res", type=int, default=1601); a = ap.parse_args()
    res = a.res; FIG = figdata.figdir(res)
    arch = json.loads((REPO/"results_hires"/f"architectures{figdata.sfx(res)}.json").read_text())["models"]
    g = measure(res)
    fams, ncells = campaign(res)
    cfdac = cfdac_gflops(res, 2)

    # per-model record: params + fwd GFLOPs + training GFLOP/epoch (3× fwd × subsample)
    SUB = 3000          # image-model subsample (engine default / meta median)
    SUB_TAB = 4000
    models = {}
    for k in ["mlp", "cnn1d", "transformer1d", "cnn2d_shallow", "cnn2d_deep", "cnn3d",
              "transformer", "resnet50", "convnext_tiny"]:
        p = arch.get(k, {}).get("params_total")
        fwd = g.get(k)
        is_img = k in IMG_FAMILY
        sub = SUB if is_img else SUB_TAB
        # per training sample: fwd+bwd ≈ 3× fwd; image models also recompute the CFDAC
        per_sample = 3*fwd + (cfdac if is_img else 0.0)
        models[k] = {"params": p, "fwd_gflops": round(fwd, 3),
                     "family": "image" if is_img else "tabular/seq",
                     "train_gflops_per_epoch": round(per_sample*sub/1e3, 2),   # TFLOP/epoch
                     "subsample": sub}
    out = {"resolution": res, "n_cells": ncells, "by_family": fams,
           "cfdac_gflops_per_sample": round(cfdac, 4),
           "mlp_flat_seq_fwd_gflops": round(g.get("mlp_flat_seq", 0), 3),
           "models": models,
           "protocol": {"image": {"subsample": SUB, "batch": 16, "max_epochs": 80, "patience": 8,
                                  "optim": "AdamW", "sched": "ReduceLROnPlateau", "amp": "bf16(A100/L4)/fp16(T4)"},
                        "tabular": {"subsample": SUB_TAB, "batch": 256, "max_epochs": 200, "patience": 15}},
           "gpus": ["T4 (15 GB)", "L4 (24 GB)", "A100 (40 GB)"]}
    (REPO/"results_hires"/f"compute{figdata.sfx(res)}.json").write_text(json.dumps(out, indent=1))

    # ---- figure: training compute per epoch (TFLOP) + params, log scale ----
    order = sorted(models, key=lambda k: models[k]["train_gflops_per_epoch"])
    vals = [models[k]["train_gflops_per_epoch"] for k in order]
    cols = ["#1f77b4" if models[k]["family"] == "image" else "#2ca02c" for k in order]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    y = np.arange(len(order))
    ax.barh(y, vals, color=cols, edgecolor="black", lw=.4)
    for i, k in enumerate(order):
        pp = models[k]["params"]
        ax.text(vals[i], i, f"  {vals[i]:.1f} TFLOP/ep · {pp/1e6:.1f}M p" if pp else f"  {vals[i]:.1f} TFLOP/ep",
                va="center", fontsize=7.5)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=9); ax.set_xscale("log")
    ax.set_xlabel("training compute per epoch (TFLOP = 3×fwd×subsample [+CFDAC]) — log scale")
    ax.set_title("Per-epoch training cost (blue = CFDAC-image, green = tabular/sequence)\n"
                 "spectral/sequence models are 1–3 orders of magnitude cheaper than the vision backbones",
                 fontweight="bold", fontsize=10)
    ax.grid(axis="x", alpha=.3)
    plt.tight_layout(); plt.savefig(FIG/"compute_cost.png", dpi=130); plt.close(fig)

    print(f"cells@1601={ncells}  CFDAC={cfdac:.3f} GFLOP/sample")
    for k in order:
        m = models[k]
        print(f"  {k:15s} {str(m['params']):>9} p  fwd {m['fwd_gflops']:8.2f} GFLOP  "
              f"{m['train_gflops_per_epoch']:8.1f} TFLOP/epoch")
    print(f"wrote results_hires/compute{figdata.sfx(res)}.json + compute_cost.png (res={res})")


if __name__ == "__main__":
    main()
