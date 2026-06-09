"""Architecture introspection for the hi-res model zoo.

Instantiates every model in representative configurations and records exact
parameter counts, a per-top-level-module breakdown, and the design schedule,
into results_hires/architectures.json. Also renders a parameter-count figure.
This is what feeds the 'Model architectures' section of the report — numbers
are measured from the actual nn.Modules, not quoted from memory.

Run: python ml_pipeline/hires_arch.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import sys
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from ml_pipeline.hires_zoo import (DeepCFDACNet, ShallowCNN2D, CNN3D, CFDACTransformer)
from ml_pipeline.hires_tab import MLP, CNN1D, Transformer1D

FIG = _REPO/"results"/"figures"/"hires"; FIG.mkdir(parents=True, exist_ok=True)


def count(m):
    tot = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return tot, tr


def by_module(m):
    return {name: int(sum(p.numel() for p in child.parameters()))
            for name, child in m.named_children()}


def fmt(n):
    return f"{n/1e6:.2f} M" if n >= 1e5 else f"{n/1e3:.1f} k"


# representative configs (n_in=2 channels e.g. cfdac_realimag; binary head)
TORCH = {}

def add(key, module, family, note, repr_cfg):
    tot, tr = count(module)
    TORCH[key] = {"family": family, "params_total": int(tot), "params_trainable": int(tr),
                  "by_module": by_module(module), "note": note, "repr_cfg": repr_cfg}
    print(f"{key:16s} {fmt(tot):>10s}  ({family})")


def main():
    import argparse
    from ml_pipeline import figdata
    global FIG
    ap = argparse.ArgumentParser(); ap.add_argument("--res", type=int, default=1601); a = ap.parse_args()
    res = a.res; FIG = figdata.figdir(res)
    arch = {"models": {}, "notes": {}}

    # ---- bespoke CFDAC-image models (n_in=2, n_out=2, full 1601 grid) ----
    add("cnn2d_shallow", ShallowCNN2D(2, 2), "CFDAC image (bespoke CNN)",
        "Port of the 128² baseline: 7×7 stride-4 stem then 3×(5×5 conv→BN→GELU→2× maxpool), "
        "widths 16→32→64; global-avg-pool → 64-d FC → logits. Deliberately shallow/cheap.",
        "n_in=2, widths=(16,32,64)")
    add("cnn2d_deep", DeepCFDACNet(2, 2), "CFDAC image (bespoke CNN)",
        "ResNet18-style: 7×7 stride-2 stem + 3×3 maxpool, then 4 stages of 2 residual BasicBlocks "
        "(two 3×3 convs + GELU + 1×1 projection shortcut), widths 64→128→256→512, each stage /2; "
        "global-avg-pool → 128-d FC (dropout 0.3) → logits. ~7 spatial downsamples digest the full 1601² grid.",
        "n_in=2, widths=(64,128,256,512)")
    add("cnn3d", CNN3D(2, 2), "CFDAC volume (bespoke 3-D CNN)",
        "Treats the CFDAC channel axis as a depth dimension: input (B,C,N,N)→(B,1,C,N,N). "
        "3-D conv stem (kernel (min(3,C),7,7), spatial stride 4) then 3×(1,3,3) stride-(1,2,2) "
        "3-D convs, widths 16→32→64; global-avg-pool3d → 64-d FC → logits.",
        "n_in=2, widths=(16,32,64)")
    add("transformer", CFDACTransformer(2, 2, input_size=res), "CFDAC image (bespoke ViT)",
        "Conv tokeniser (5 strided convs, total /64: 1601→~25) → ~625 tokens of dim 192; prepend a "
        "CLS token + learned positional embedding; 6-layer pre-norm TransformerEncoder (6 heads, MLP "
        "ratio 4, GELU, dropout 0.1); LayerNorm → linear head on the CLS token. Tokenises the full-res "
        "CFDAC rather than resizing to 224.",
        "n_in=2, dim=192, depth=6, heads=6, input_size=%d"%res)

    # ---- tabular / sequence models ----
    add("mlp", MLP(81, 2), "tabular / flattened (MLP)",
        "Fully-connected: 3 hidden layers 512→256→128, each Linear→BatchNorm1d→GELU→Dropout(0.3), "
        "then a linear head. Shown for modal (d_in=81); d_in = feature length (22 for indicators, "
        "C×L for a flattened sequence, e.g. 9×1601=14 409 for frf_mag).",
        "d_in=81 (modal), hidden=(512,256,128)")
    add("cnn1d", CNN1D(9, 2), "sequence over frequency (1-D CNN)",
        "1-D CNN over the frequency/time axis: 7-wide stride-2 stem then 3×(5-wide stride-2 conv→BN→GELU), "
        "widths 32→64→128 (each /2); global-avg-pool1d → 64-d FC → logits. Shown for a 9-channel input "
        "(frf_mag / timeseries); 18 channels for frf_realimag.",
        "c_in=9, widths=(32,64,128)")
    add("transformer1d", Transformer1D(9, 2, length=1601), "sequence over frequency (1-D ViT)",
        "Conv tokeniser (15-wide /8 then 5-wide /4, total /32) → tokens of dim 128; CLS + learned pos-embed; "
        "4-layer pre-norm TransformerEncoder (4 heads, MLP ratio 4, GELU, dropout 0.1); LayerNorm → linear "
        "head on CLS. Shown for 9 channels × length 1601.",
        "c_in=9, dim=128, depth=4, heads=4, length=1601")

    arch["models"].update(TORCH)

    # ---- timm vision backbones (ImageNet-pretrained, adapted) ----
    try:
        import timm
        for key in ("resnet50", "convnext_tiny"):
            net = timm.create_model(key, pretrained=False, in_chans=2, num_classes=2)
            tot, tr = count(net)
            cfg = getattr(net, "default_cfg", {}) or {}
            arch["models"][key] = {
                "family": "CFDAC image (ImageNet-pretrained backbone)",
                "params_total": int(tot), "params_trainable": int(tr),
                "by_module": {}, "repr_cfg": "in_chans=2, num_classes=2, pretrained=ImageNet-1k",
                "note": ("ResNet50: classic 4-stage bottleneck CNN (25.6 M params), fed CFDAC images "
                         "resized to 384²." if key == "resnet50" else
                         "ConvNeXt-Tiny: modern CNN (depthwise 7×7 + inverted bottleneck, 28 M params), "
                         "fed CFDAC images at native conv resolution.")
                + " ImageNet weights loaded, the input stem adapted to the requested channel count "
                  "(in_chans=2 here) and the classifier head replaced for the task. Warm-up: head-only "
                  "for 2 epochs (LR 3e-4), then the whole backbone is unfrozen at LR 3e-5."}
            print(f"{key:16s} {fmt(tot):>10s}  (timm)")
    except Exception as e:
        print("timm introspection skipped:", e)

    # ---- non-parametric models ----
    arch["models"]["rf"] = {"family": "tabular (ensemble)", "params_total": None,
        "note": "RandomForest, 400 trees, class_weight='balanced' (cls) / plain (reg), all CPU cores. "
                "Non-parametric — 'size' is the forest, not a weight count. On modal(81)/indicators(22) only.",
        "repr_cfg": "n_estimators=400, class_weight='balanced'"}
    arch["models"]["xgb"] = {"family": "tabular (gradient boosting)", "params_total": None,
        "note": "XGBoost, 600 trees, max_depth=6, lr=0.05, subsample=0.8, colsample_bytree=0.8; "
                "multi:softprob / binary:logistic. On modal(81)/indicators(22) only.",
        "repr_cfg": "n_estimators=600, max_depth=6, lr=0.05"}

    (_REPO/"results_hires"/f"architectures{figdata.sfx(res)}.json").write_text(json.dumps(arch, indent=1))

    # ---- parameter-count figure (torch + timm models) ----
    items = [(k, v["params_total"]) for k, v in arch["models"].items() if v.get("params_total")]
    items.sort(key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = [k for k, _ in items]; vals = [v/1e6 for _, v in items]
    cols = ["#1f77b4" if arch["models"][k]["family"].startswith("CFDAC") else "#2ca02c" for k in names]
    ax.barh(np.arange(len(names)), vals, color=cols, edgecolor="black", lw=.4)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.2f}M", va="center", fontsize=8)
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("trainable parameters (millions)")
    ax.set_xscale("log")
    ax.set_title("Model capacity (blue = CFDAC-image, green = tabular/sequence)\n"
                 "pretrained vision backbones are 10–100× larger yet do not win", fontweight="bold", fontsize=10)
    ax.grid(axis="x", alpha=.3)
    plt.tight_layout(); plt.savefig(FIG/"arch_params.png", dpi=130); plt.close(fig)
    print(f"wrote results_hires/architectures{figdata.sfx(res)}.json + arch_params.png (res={res})")


if __name__ == "__main__":
    main()
