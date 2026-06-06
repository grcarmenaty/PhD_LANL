"""Hi-res 1601 CFDAC model zoo — shared engine for the Colab GPU notebooks.

One place that defines:
  * the CFDAC channel-features (real / imag / mag / phase and their stacks),
  * GPU CFDAC (the exact pymodal value_CFDAC formula, in torch),
  * every model family — bespoke CNNs (shallow + deep), a 3-D CNN, a
    conv-tokenised Transformer, and timm vision backbones,
  * a generic train+evaluate routine that writes per-case JSON with the
    SAME schema as results_hires/ (so hires_summary.py / the report ingest
    both), skip-if-exists, best-val checkpointing, AMP, cosine schedule.

The notebooks (notebooks/hires_*_gpu.ipynb) just set a CELLS grid + an
output dir (Google Drive for persistence) and call run_cells(...).

CPU-importable and CPU-runnable (slow); designed for CUDA.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# A100-friendly defaults: TF32 matmul/conv (big speedup, negligible accuracy
# loss on the CFDAC matmul + convs) and high-precision matmul path.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass


def _amp_dtype(dev):
    """bfloat16 on A100/H100 (no GradScaler needed); float16 elsewhere."""
    if dev.type != "cuda":
        return torch.float32
    try:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    except Exception:
        return torch.float16

# --------------------------------------------------------------------------
# CFDAC features (channel sets) and the GPU CFDAC transform
# --------------------------------------------------------------------------

CFDAC_FEATURES: Dict[str, Tuple[str, ...]] = {
    "cfdac_real":     ("real",),
    "cfdac_imag":     ("imag",),
    "cfdac_mag":      ("mag",),
    "cfdac_phase":    ("phase",),
    "cfdac_realimag": ("real", "imag"),
    "cfdac_magphase": ("mag", "phase"),
    "cfdac_all":      ("real", "imag", "mag", "phase"),
}

VISION_BACKBONES = ("resnet50", "convnext_tiny", "efficientnet_b0",
                    "swin_tiny_patch4_window7_224", "vit_base_patch16_224")
BESPOKE_MODELS   = ("cnn2d_deep", "cnn2d_norm", "cnn2d_shallow", "cnn3d", "transformer")
ALL_MODELS       = BESPOKE_MODELS + VISION_BACKBONES
# Vision/Swin/ViT have a fixed input size; conv backbones can take any size.
_FIXED_224 = ("swin_tiny_patch4_window7_224", "vit_base_patch16_224")


def cfdac_torch(ref: torch.Tensor, frf: torch.Tensor,
                channels: Sequence[str], normalize: bool = True) -> torch.Tensor:
    """ref:(N,9) complex shared; frf:(B,N,9) complex -> (B,C,N,N) float32.

    Matches pymodal.utils.value_CFDAC:
      C[i,j] = (frf[i]·conj(ref[j]))^2 / (||frf[i]||^2 ||ref[j]||^2)
    """
    with torch.no_grad():
        cross = torch.matmul(frf, ref.conj().t())               # (B,N,N)
        num   = cross ** 2
        nf    = (frf.abs() ** 2).sum(-1)                         # (B,N)
        nr    = (ref.abs() ** 2).sum(-1)                         # (N,)
        denom = nf.unsqueeze(-1) * nr.unsqueeze(0).unsqueeze(0)  # (B,N,N)
        C = torch.nan_to_num(num / denom)                        # complex (B,N,N)
        outs = []
        for ch in channels:
            if   ch == "real":  v = C.real.float()
            elif ch == "imag":  v = C.imag.float()
            elif ch == "mag":   v = C.abs().float()
            elif ch == "phase": v = torch.angle(C).float()
            else: raise ValueError(ch)
            if normalize:
                if ch in ("real", "imag"):
                    v = v - v.mean(dim=(-2, -1), keepdim=True)
                elif ch == "mag":
                    v = 2.0 * v - 1.0; v = v - v.mean(dim=(-2, -1), keepdim=True)
                elif ch == "phase":
                    v = v / np.float32(np.pi)
            outs.append(v)
        return torch.stack(outs, dim=1)                          # (B,C,N,N)


# --------------------------------------------------------------------------
# Bespoke models
# --------------------------------------------------------------------------

def _conv_bn(ci, co, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(ci, co, k, s, p, bias=False), nn.BatchNorm2d(co))


class _BasicBlock(nn.Module):
    def __init__(self, ci, co, stride=1):
        super().__init__()
        self.c1 = _conv_bn(ci, co, 3, stride, 1); self.c2 = _conv_bn(co, co, 3, 1, 1)
        self.act = nn.GELU()
        self.down = _conv_bn(ci, co, 1, stride, 0) if (stride != 1 or ci != co) else None
    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        x = self.act(self.c1(x)); x = self.c2(x)
        return self.act(x + idt)


class DeepCFDACNet(nn.Module):
    """ResNet18-style; consumes the full 1601 grid (7 downsamples)."""
    def __init__(self, n_in=2, n_out=2, widths=(64, 128, 256, 512), regression=False):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(n_in, widths[0], 7, 2, 3, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.GELU(), nn.MaxPool2d(3, 2, 1))
        c = widths[0]; stages = []
        for i, w in enumerate(widths):
            s = 1 if i == 0 else 2
            stages += [_BasicBlock(c, w, s), _BasicBlock(w, w, 1)]; c = w
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(c, 128), nn.GELU(), nn.Dropout(0.3),
                                  nn.Linear(128, n_out))
        self.regression = regression
    def forward(self, x): return self.head(self.stages(self.stem(x)))


class ShallowCNN2D(nn.Module):
    """Port of the 128-baseline Conv2DStack (strided stem + 3 conv/pool)."""
    def __init__(self, n_in=2, n_out=2, widths=(16, 32, 64), regression=False):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(n_in, widths[0], 7, 4, 3),
                                  nn.BatchNorm2d(widths[0]), nn.GELU())
        c = widths[0]; layers = []
        for w in widths:
            layers += [nn.Conv2d(c, w, 5, padding=2), nn.BatchNorm2d(w), nn.GELU(), nn.MaxPool2d(2)]; c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(c, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, n_out))
        self.regression = regression
    def forward(self, x): return self.head(self.features(self.stem(x)))


class MidCNN2D(nn.Module):
    """Medium ("normal") plain 2-D CNN — between ShallowCNN2D and the deep
    ResNet. stride-2 stem (keeps more resolution than shallow's stride-4) + 4
    Conv/BN/GELU/MaxPool blocks (32->64->128->256), no residuals."""
    def __init__(self, n_in=2, n_out=2, widths=(32, 64, 128, 256), regression=False):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(n_in, widths[0], 7, 2, 3),
                                  nn.BatchNorm2d(widths[0]), nn.GELU(), nn.MaxPool2d(2))
        c = widths[0]; layers = []
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, padding=1), nn.BatchNorm2d(w), nn.GELU(), nn.MaxPool2d(2)]; c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(c, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, n_out))
        self.regression = regression
    def forward(self, x): return self.head(self.features(self.stem(x)))


class CNN3D(nn.Module):
    """Treats CFDAC channels as a depth axis: input (B,C,N,N) -> (B,1,C,N,N)."""
    def __init__(self, n_in=2, n_out=2, widths=(16, 32, 64), regression=False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, widths[0], (min(3, n_in), 7, 7), (1, 4, 4), (min(3, n_in) // 2, 3, 3)),
            nn.BatchNorm3d(widths[0]), nn.GELU())
        c = widths[0]; layers = []
        for w in widths:
            layers += [nn.Conv3d(c, w, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
                       nn.BatchNorm3d(w), nn.GELU()]; c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                                  nn.Linear(c, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, n_out))
        self.regression = regression
    def forward(self, x): return self.head(self.features(self.stem(x.unsqueeze(1))))


class CFDACTransformer(nn.Module):
    """Conv tokeniser (strided convs) -> tokens -> Transformer encoder -> cls head.
    Tokenises the full-res CFDAC instead of resizing to 224."""
    def __init__(self, n_in=2, n_out=2, input_size=1601, dim=192, depth=6, heads=6,
                 regression=False):
        super().__init__()
        self.tok = nn.Sequential(
            nn.Conv2d(n_in, 64, 7, 4, 3), nn.BatchNorm2d(64), nn.GELU(),    # /4
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.GELU(),    # /8
            nn.Conv2d(128, dim, 3, 2, 1), nn.BatchNorm2d(dim), nn.GELU(),   # /16
            nn.Conv2d(dim, dim, 3, 2, 1), nn.BatchNorm2d(dim), nn.GELU(),   # /32
            nn.Conv2d(dim, dim, 3, 2, 1), nn.BatchNorm2d(dim), nn.GELU(),   # /64
        )
        with torch.no_grad():
            g = self.tok(torch.zeros(1, n_in, input_size, input_size)).shape[-1]
        self.n_tok = g * g
        self.dim = dim
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.n_tok + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.1,
                                         activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_out)
        self.regression = regression
    def forward(self, x):
        t = self.tok(x).flatten(2).transpose(1, 2)                 # (B, n_tok, dim)
        b = t.shape[0]
        t = torch.cat([self.cls.expand(b, -1, -1), t], dim=1) + self.pos
        t = self.norm(self.enc(t))
        return self.head(t[:, 0])


def build_model(model: str, n_in: int, n_out: int, kind: str,
                input_size: int = 1601, vision_size: int = 384,
                pretrained: bool = True):
    """Returns (module, feed_size) where feed_size is the spatial size the
    model wants (None = use full input_size as-is)."""
    reg = (kind == "reg")
    if model == "cnn2d_deep":    return DeepCFDACNet(n_in, n_out, regression=reg), None
    if model == "cnn2d_norm":    return MidCNN2D(n_in, n_out, regression=reg), None
    if model == "cnn2d_shallow": return ShallowCNN2D(n_in, n_out, regression=reg), None
    if model == "cnn3d":         return CNN3D(n_in, n_out, regression=reg), None
    if model == "transformer":   return CFDACTransformer(n_in, n_out, input_size, regression=reg), None
    if model in VISION_BACKBONES:
        import timm
        size = 224 if model in _FIXED_224 else vision_size
        net = timm.create_model(model, pretrained=pretrained, in_chans=n_in,
                                num_classes=n_out, img_size=size) \
            if model in _FIXED_224 else \
            timm.create_model(model, pretrained=pretrained, in_chans=n_in, num_classes=n_out)
        net.regression = reg
        return net, size
    raise ValueError(f"unknown model {model!r}")


# --------------------------------------------------------------------------
# Train + evaluate one cell
# --------------------------------------------------------------------------

def _resize(x, size):
    if size is None or x.shape[-1] == size: return x
    return nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def run_cell(task: str, model: str, feature: str, *, syn_h5, exp_h5, out_dir: Path,
             dev, syn_tasks, exp_tasks, H_ref_syn, H_ref_exp, H_exp, exp_names,
             make_split, epochs=25, subsample=3000, batch=16, lr=3e-4, warmup=2,
             vision_size=384, seed=42, pretrained=True, force=False, log=print,
             max_epochs=80, patience=8, min_delta=1e-3, ckpt_every=5):
    """One cell, trained ONCE to convergence with per-epoch checkpointing.

    Resilience / convergence:
      * skip-if-exists on the per-case JSON  -> a finished cell is never redone;
      * a per-epoch checkpoint on Drive (out_dir/models/<tag>.ckpt)            ->
        a session cut off mid-cell RESUMES from the last epoch, so each cell is
        trained exactly once even across disconnects;
      * early stopping (val macro-F1 / R2, `patience` epochs, `min_delta`) +
        ReduceLROnPlateau, capped at `max_epochs`  -> trained to convergence,
        not a fixed epoch count. (`epochs` kept only for back-compat; ignored.)
    """
    import h5py
    from sklearn.metrics import (accuracy_score, f1_score, r2_score,
                                 balanced_accuracy_score)
    out_dir = Path(out_dir); (out_dir / "per_case").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    channels = CFDAC_FEATURES[feature]; n_in = len(channels)
    tag = f"{task}_{model}_{feature}_hires1601"
    pc = out_dir / "per_case" / f"{tag}.json"
    if pc.exists() and not force:
        log(f"skip {tag} (exists)"); return
    is_vision = model in VISION_BACKBONES

    mask, y_all, kind = syn_tasks[task]
    idx_pool = np.where(mask)[0]
    rng = np.random.default_rng(20260518 + seed)
    keep = np.sort(rng.choice(len(idx_pool), size=min(subsample, len(idx_pool)), replace=False))
    y = y_all[keep]
    abs_idx = idx_pool[keep]; order = np.argsort(abs_idx); sr = abs_idx[order]
    with h5py.File(syn_h5, "r") as f:
        re = f["frf_real"][sr]; im = f["frf_imag"][sr]
    H = np.empty_like(re, dtype=np.complex64); H[order] = (re + 1j * im).astype(np.complex64)
    i_tr, i_va, i_te = make_split(y, kind); n_out = (int(y.max()) + 1) if kind == "cls" else 1
    # Keep the synth subsample FRFs resident on the GPU (≈0.35 GB) so no
    # host->device copy happens per batch — the A100 then never waits on the host.
    Hg = torch.from_numpy(H).to(dev)
    yg = torch.from_numpy(y).to(dev) if kind == "cls" else torch.from_numpy(y.astype(np.float32)).to(dev)

    net, feed = build_model(model, n_in, n_out, kind, vision_size=vision_size,
                            pretrained=pretrained)
    net = net.to(dev)
    cls_w = None
    if kind == "cls":
        cnt = np.bincount(y[i_tr], minlength=n_out).astype(np.float32)
        cls_w = torch.tensor(cnt.sum() / np.clip(cnt * n_out, 1e-6, None)).float().to(dev)
    lossf = nn.CrossEntropyLoss(weight=cls_w) if kind == "cls" else nn.MSELoss()
    amp_dt = _amp_dtype(dev)
    use_amp = (dev.type == "cuda")
    # GradScaler only needed for fp16; bf16/fp32 run without it.
    need_scaler = use_amp and amp_dt == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=need_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=need_scaler)

    # vision: head-only for the first `warmup` epochs, then unfreeze backbone.
    def set_frozen(frozen: bool):
        if not is_vision:
            return
        for n_, p in net.named_parameters():
            p.requires_grad = (not frozen) or any(
                k in n_ for k in ("fc", "classifier", "head", "norm"))

    def build_opt(ep):
        """Optimizer/scheduler for the regime at epoch `ep` (rebuilt fresh on
        resume; LR/momentum reset is acceptable for a rare disconnect)."""
        unfrozen = (not is_vision) or (ep >= warmup)
        set_frozen(frozen=not unfrozen)
        base_lr = lr if not (is_vision and unfrozen) else lr * 0.1
        params = [p for p in net.parameters() if p.requires_grad]
        o = torch.optim.AdamW(params, lr=base_lr, weight_decay=1e-4)
        mode = "max"   # macro-F1 and R2 are both higher-is-better
        s = torch.optim.lr_scheduler.ReduceLROnPlateau(o, mode=mode, factor=0.5,
                                                       patience=3, min_lr=1e-6)
        return o, s

    def predict(Hsrc, ref, kk, bs=batch):
        """Hsrc is a GPU complex tensor (n, 1601, 9)."""
        net.eval(); preds = []; probs = []
        with torch.no_grad():
            for i in range(0, len(Hsrc), bs):
                fb = Hsrc[i:i + bs]
                x = _resize(cfdac_torch(ref, fb, channels), feed)
                with torch.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                    out = net(x).float().cpu().numpy()
                if kk == "cls":
                    preds.append(out.argmax(1))
                    e = np.exp(out - out.max(1, keepdims=True)); probs.append(e / e.sum(1, keepdims=True))
                else:
                    preds.append(out.squeeze(1) if out.ndim == 2 else out)
        return np.concatenate(preds), (np.concatenate(probs) if kk == "cls" else None)

    # ---- resume from a mid-cell checkpoint on Drive, if one exists ----
    ckpt_path = out_dir / "models" / f"{tag}.ckpt"
    best_path = out_dir / "models" / f"{tag}.best.pt"
    start_ep = 0; best = -1e9; since = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev)
        net.load_state_dict(ck["model"]); start_ep = ck["epoch"]
        best = ck["best"]; since = ck["since"]
        log(f"    {tag}: resume from epoch {start_ep} (best={best:+.4f}, since={since})")

    opt, sched = build_opt(start_ep)
    t0 = time.time()
    for ep in range(start_ep, max_epochs):
        if (is_vision and ep == warmup):
            opt, sched = build_opt(ep)            # unfreeze regime
            log(f"    {tag} ep{ep+1}: unfreeze backbone")
        net.train(); perm = np.random.permutation(i_tr)
        for j in range(0, len(perm), batch):
            bi = perm[j:j + batch]
            fb = Hg[bi]
            x = _resize(cfdac_torch(H_ref_syn, fb, channels), feed)
            yb = yg[bi]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                out = net(x)
                loss = lossf(out, yb.long()) if kind == "cls" else lossf(out, yb.float().unsqueeze(1))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        pv, _ = predict(Hg[i_va], H_ref_syn, kind)
        m = (f1_score(y[i_va], pv, labels=list(range(n_out)), average="macro", zero_division=0)
             if kind == "cls" else r2_score(y[i_va], pv))
        sched.step(m)
        improved = m > best + min_delta
        if improved:
            best = float(m); since = 0
            torch.save({k: v.detach().clone() for k, v in net.state_dict().items()}, best_path)
        else:
            since += 1
        # checkpoint to Drive every `ckpt_every` epochs (+ at stop) — not every
        # epoch, so Drive I/O doesn't stall the GPU. Resume loses <= ckpt_every epochs.
        if ((ep + 1) % ckpt_every == 0) or (ep >= warmup and since >= patience):
            torch.save({"model": {k: v.detach().clone() for k, v in net.state_dict().items()},
                        "epoch": ep + 1, "best": best, "since": since}, ckpt_path)
        log(f"    {tag} ep{ep+1}/{max_epochs} val={m:+.4f} best={best:+.4f} "
            f"since={since} ({time.time()-t0:.0f}s)")
        if ep >= warmup and since >= patience:
            log(f"    {tag}: converged (no +{min_delta} in {patience} epochs)"); break
    # restore best weights
    if best_path.exists():
        net.load_state_dict(torch.load(best_path, map_location=dev))
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, out_dir / "models" / f"{tag}.pt")

    pt, _ = predict(Hg[i_te], H_ref_syn, kind)
    if kind == "cls":
        s_acc = accuracy_score(y[i_te], pt)
        s_mf1 = f1_score(y[i_te], pt, labels=list(range(n_out)), average="macro", zero_division=0)
        s_bal = balanced_accuracy_score(y[i_te], pt)
    else:
        s_acc = r2_score(y[i_te], pt); s_mf1 = None; s_bal = None

    mask_e, e_y, e_kind = exp_tasks[task]; idx_e = np.where(mask_e)[0]
    He = torch.from_numpy(H_exp[idx_e]).to(dev)            # exp FRFs -> GPU once
    pe, prob = predict(He, H_ref_exp, e_kind)
    del He
    if dev.type == "cuda": torch.cuda.empty_cache()
    rows = []
    for i, ix in enumerate(idx_e):
        r = {"case": exp_names[ix],
             "y_true": (int(e_y[i]) if e_kind == "cls" else float(e_y[i])),
             "y_pred": (int(pe[i]) if e_kind == "cls" else float(pe[i]))}
        if prob is not None: r["proba"] = [float(p) for p in prob[i]]
        rows.append(r)
    meta = {"task": task, "model": model, "feature": feature, "n_target": 1601,
            "n_channels": n_in, "kind": e_kind, "n_out": int(n_out),
            "max_epochs": max_epochs, "patience": patience,
            "subsample": int(min(subsample, len(idx_pool))),
            "vision_feed_size": feed, "synth_val_best": best,
            "synth_test_metric": float(s_acc),
            "synth_test_macro_f1": (float(s_mf1) if s_mf1 is not None else None),
            "synth_test_bal_acc": (float(s_bal) if s_bal is not None else None),
            "input_normalized": True, "trained_to_convergence": True}
    pc.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    sp = out_dir / "synth_test_zoo.json"
    allr = json.loads(sp.read_text()) if sp.exists() else {}
    allr[tag] = meta; sp.write_text(json.dumps(allr, indent=2))
    # cell finished -> drop the resume checkpoint + best snapshot (per_case is the done flag)
    for _p in (ckpt_path, best_path):
        try: _p.unlink()
        except FileNotFoundError: pass
    log(f"  DONE {tag}: synth {'mF1' if kind=='cls' else 'R2'}="
        f"{(s_mf1 if s_mf1 is not None else s_acc):.3f}  -> {pc.name}")


def all_cfdac_cells(models: Sequence[str], tasks: Sequence[str],
                    features: Sequence[str] = tuple(CFDAC_FEATURES)) -> List[Tuple[str, str, str]]:
    """Full (task, model, feature) grid over CFDAC features."""
    return [(t, m, f) for t in tasks for m in models for f in features]
