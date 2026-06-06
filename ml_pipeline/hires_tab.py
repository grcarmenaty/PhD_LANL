"""Hi-res 1601 NON-image feature zoo — the complement to hires_zoo.py.

Covers the feature families that are NOT CFDAC images — the ones that actually
transferred best in the 128 baseline (modal / indicators), plus raw FRF:

  features (computed from the 1601-bin FRFs in features_hires.h5 / _exp):
    modal         (81)   physics peaks/energy per channel (modal_features)
    indicators    (22)   pymodal damage indicators vs pristine ref
    frf_mag       (9,1601)  per-sample log-normalised |H|  (sequence)
    frf_realimag  (18,1601) per-sample normalised Re/Im H  (sequence)

  models:
    mlp            proper MLP (BN+GELU+dropout) on vec / flattened seq
    cnn1d          1-D CNN over frequency (sequence features)
    transformer1d  conv-tokenised 1-D transformer (sequence features)
    rf             RandomForest (sklearn)        on vec features
    xgb            XGBoost (if installed)        on vec features

Scientifically-sound protocol (matches the project's findings):
  * fixed 70/15/15 split (make_split), class-weighted losses / balanced trees;
  * features standardised with stats fit on the SYNTH-TRAIN fold only, then
    applied to synth-test and the experimental set -> honest zero-shot transfer
    (sequence features use per-sample normalisation, which needs no fit);
  * NN models train to convergence (early stop + ReduceLROnPlateau) with
    per-epoch checkpoint/resume; trees fit once (skip-if-exists handles resume);
  * reported with macro-F1 / balanced-accuracy (cls) and R2/MAE (reg), same
    per-case JSON schema as results_hires/ so hires_summary + the report ingest it.

A per-sample feature CACHE (.npy on Drive) is built once for ALL samples, so
indicators (which recompute a 1601 CFDAC per sample) are not recomputed per task.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# feature -> kind ('vec' tabular | 'seq' (C,L))
TAB_FEATURES = {"modal": "vec", "indicators": "vec",
                "frf_mag": "seq", "frf_realimag": "seq", "timeseries": "seq"}
# model -> compatible features
TAB_MODEL_FEATURES = {
    "mlp":           ["modal", "indicators", "frf_mag", "frf_realimag", "timeseries"],
    "rf":            ["modal", "indicators"],
    "xgb":           ["modal", "indicators"],
    "cnn1d":         ["frf_mag", "frf_realimag", "timeseries"],
    "transformer1d": ["frf_mag", "frf_realimag", "timeseries"],
}
NN_MODELS = ("mlp", "cnn1d", "transformer1d")


def _amp_dtype(dev):
    if dev.type != "cuda": return torch.float32
    try: return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    except Exception: return torch.float16


# --------------------------------------------------------------------------
# Feature extraction (whole-dataset cache)
# --------------------------------------------------------------------------

def _seq_normalise(seq: np.ndarray) -> np.ndarray:
    """Per-sample, per-channel z-score over the frequency axis. seq:(n,C,L)."""
    mu = seq.mean(axis=2, keepdims=True); sd = seq.std(axis=2, keepdims=True) + 1e-6
    return ((seq - mu) / sd).astype(np.float32)


def _chirp_spectrum(n_bins: int, n_t: int = 4096, fs: float = 256.0,
                    f_lo: float = 2.0, f_hi: float = 100.0) -> np.ndarray:
    """rfft of the deterministic excitation chirp, truncated to `n_bins`
    (the stored 0–100 Hz band). Amplitude is irrelevant (per-sample z-score
    removes scale), so we use amplitude 1. Matches generate_dataset.make_chirp
    with the hires N_T=4096, fs=256 generation parameters."""
    t = np.arange(n_t) / fs
    k = (f_hi - f_lo) / (n_t / fs)
    chirp = np.sin(2 * np.pi * (f_lo * t + 0.5 * k * t * t))
    return np.fft.rfft(chirp)[:n_bins]            # (n_bins,) complex


def _timeseries_from_frf(re: np.ndarray, im: np.ndarray, freqs: np.ndarray,
                         n_t: int = 4096) -> np.ndarray:
    """Reconstruct the band-limited time response from the stored 0–100 Hz FRF,
    IDENTICALLY for synth and exp: ts = irfft(H · chirp_spectrum). Returns
    (n, 9, n_t) per-sample-normalised. This is FRF-derived (the experimental set
    has no measured timeseries), so synth and exp share the exact same pipeline
    and the only difference is the FRF content itself."""
    H = (re + 1j * im).astype(np.complex64)        # (n, nb, 9)
    nb = H.shape[1]; n_full = n_t // 2 + 1
    cs = _chirp_spectrum(nb).astype(np.complex64)  # (nb,)
    resp = np.zeros((H.shape[0], n_full, H.shape[2]), dtype=np.complex64)
    resp[:, :nb, :] = H * cs[None, :, None]
    ts = np.fft.irfft(resp, n=n_t, axis=1)         # (n, n_t, 9)
    return _seq_normalise(ts.transpose(0, 2, 1).astype(np.float32))   # (n,9,n_t)


def _decf(a, res):
    """Bin-average the frequency axis (axis=1) of a:(n,nfreq,...) down to `res`."""
    n = a.shape[1]
    if res >= n:
        return a
    edges = np.linspace(0, n, res + 1).astype(int); starts = edges[:-1]
    s = np.add.reduceat(a, starts, axis=1)
    sizes = np.maximum(np.diff(edges), 1).reshape((1, res) + (1,) * (a.ndim - 2))
    return (s / sizes).astype(a.dtype)


def build_feature_cache(h5_path, feature: str, cache_path: Path, log=print, res=1601):
    """Compute `feature` for ALL samples in `h5_path` at frequency resolution
    `res` (<=1601), cache to .npy, return it.

    Robust to Google-Drive FUSE failures on large files: if the requested
    (Drive) path can't be written, falls back to local /content/cache, and in
    any case returns the in-memory array so the run never blocks.
    """
    import h5py, os
    cache_path = Path(cache_path)
    local = Path("/content/cache") / cache_path.name
    for p in (cache_path, local):
        if p.exists():
            try: return np.load(p, mmap_mode="r")
            except Exception: pass
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ml_pipeline.features import modal_features, indicator_features
    with h5py.File(h5_path, "r") as f:
        freqs = _decf(f["freqs"][:].astype(np.float32)[None, :, None], res)[0, :, 0]
        H_ref = _decf(f["reference/frf_complex"][:][None], res)[0]            # (res,9)
        n = f["frf_real"].shape[0]
        if feature == "frf_mag":
            mag = _decf(np.abs(f["frf_real"][:] + 1j * f["frf_imag"][:]).astype(np.float32), res)
            X = _seq_normalise(np.log10(np.clip(mag.transpose(0, 2, 1), 1e-12, None)))
        elif feature == "frf_realimag":
            re = _decf(f["frf_real"][:], res); im = _decf(f["frf_imag"][:], res)
            X = _seq_normalise(np.concatenate([re, im], axis=2).transpose(0, 2, 1).astype(np.float32))
        elif feature == "timeseries":
            # reconstruct at full res, then decimate the TIME axis by the same ratio
            ts = _timeseries_from_frf(f["frf_real"][:], f["frf_imag"][:], freqs)  # (n,9,4096)
            tgt = max(64, int(round(ts.shape[2] * res / 1601)))
            X = _seq_normalise(_decf(ts.transpose(0, 2, 1), tgt).transpose(0, 2, 1))
        elif feature == "modal":
            mag = _decf(np.abs(f["frf_real"][:] + 1j * f["frf_imag"][:]).astype(np.float32), res)
            X = np.stack([modal_features(mag[i], freqs) for i in range(n)]).astype(np.float32)
        elif feature == "indicators":
            reA = _decf(f["frf_real"][:], res); imA = _decf(f["frf_imag"][:], res)
            X = np.empty((n, 22), np.float32)
            for i in range(n):
                X[i] = indicator_features((reA[i] + 1j * imA[i]).astype(np.complex64), H_ref)
                if (i + 1) % 1000 == 0: log(f"  indicators {i+1}/{n}")
        else:
            raise ValueError(feature)
    # save robustly: try the requested (Drive) path, fall back to local /content
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, X)
        log(f"cached {feature} {X.shape} -> {cache_path}")
    except Exception as e:
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.save(local, X)
            log(f"  Drive save failed ({type(e).__name__}); cached locally -> {local}")
        except Exception as e2:
            log(f"  cache save failed ({e2}); continuing with in-memory array only")
    return X


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, d_in, n_out, hidden=(512, 256, 128), dropout=0.3, regression=False):
        super().__init__()
        layers = []; d = d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]; d = h
        layers += [nn.Linear(d, n_out)]
        self.net = nn.Sequential(*layers); self.regression = regression
    def forward(self, x): return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, c_in, n_out, widths=(32, 64, 128), regression=False):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(c_in, widths[0], 7, 2, 3), nn.BatchNorm1d(widths[0]), nn.GELU())
        c = widths[0]; layers = []
        for w in widths:
            layers += [nn.Conv1d(c, w, 5, 2, 2), nn.BatchNorm1d(w), nn.GELU()]; c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(c, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, n_out))
        self.regression = regression
    def forward(self, x): return self.head(self.features(self.stem(x)))


class Transformer1D(nn.Module):
    def __init__(self, c_in, n_out, length=1601, dim=128, depth=4, heads=4, regression=False):
        super().__init__()
        self.tok = nn.Sequential(nn.Conv1d(c_in, dim, 15, 8, 7), nn.GELU(),
                                 nn.Conv1d(dim, dim, 5, 4, 2), nn.GELU())   # /32 tokens
        with torch.no_grad():
            ntok = self.tok(torch.zeros(1, c_in, length)).shape[-1]
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, ntok + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.1, activation="gelu",
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, depth); self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_out); self.regression = regression
    def forward(self, x):
        t = self.tok(x).transpose(1, 2); b = t.shape[0]
        t = torch.cat([self.cls.expand(b, -1, -1), t], 1) + self.pos
        return self.head(self.norm(self.enc(t))[:, 0])


def _build_tab_model(model, feature, d_in_or_cin, n_out, kind, length=1601):
    reg = (kind == "reg")
    if model == "mlp":           return MLP(d_in_or_cin, n_out, regression=reg)
    if model == "cnn1d":         return CNN1D(d_in_or_cin, n_out, regression=reg)
    if model == "transformer1d": return Transformer1D(d_in_or_cin, n_out, length=length, regression=reg)
    raise ValueError(model)


# --------------------------------------------------------------------------
# Train + evaluate one tabular/sequence cell
# --------------------------------------------------------------------------

def _metrics_cls(yt, yp, n_out):
    from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
    return (float(accuracy_score(yt, yp)),
            float(f1_score(yt, yp, labels=list(range(n_out)), average="macro", zero_division=0)),
            float(balanced_accuracy_score(yt, yp)))


def run_tab_cell(task, model, feature, *, out_dir: Path, dev, syn_tasks, exp_tasks,
                 Xsyn, Xexp, exp_names, make_split, subsample=4000, batch=256, lr=1e-3,
                 max_epochs=200, patience=15, min_delta=1e-3, seed=42, force=False, log=print,
                 ckpt_every=5, res=1601):
    """Xsyn/Xexp are the WHOLE-dataset cached feature arrays for `feature`."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    out_dir = Path(out_dir); (out_dir / "per_case").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    tag = f"{task}_{model}_{feature}_hires{res}"
    pc = out_dir / "per_case" / f"{tag}.json"
    if pc.exists() and not force:
        log(f"skip {tag} (exists)"); return
    is_seq = TAB_FEATURES[feature] == "seq"

    mask, y_all, kind = syn_tasks[task]
    idx_pool = np.where(mask)[0]
    rng = np.random.default_rng(20260518 + seed)
    keep = np.sort(rng.choice(len(idx_pool), size=min(subsample, len(idx_pool)), replace=False))
    y = y_all[keep]
    Xs = np.asarray(Xsyn[idx_pool[keep]])
    i_tr, i_va, i_te = make_split(y, kind); n_out = (int(y.max()) + 1) if kind == "cls" else 1

    mask_e, e_y, e_kind = exp_tasks[task]; idx_e = np.where(mask_e)[0]
    Xe = np.asarray(Xexp[idx_e])

    # ---- standardise (fit on synth-train only); seq already per-sample normed ----
    if not is_seq:
        sc = StandardScaler().fit(Xs[i_tr])
        Xs = sc.transform(Xs).astype(np.float32); Xe = sc.transform(Xe).astype(np.float32)
    # mlp on a sequence feature -> flatten
    if model == "mlp" and is_seq:
        Xs = Xs.reshape(len(Xs), -1); Xe = Xe.reshape(len(Xe), -1)

    t0 = time.time()
    if model in ("rf", "xgb"):
        if model == "rf":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            m = (RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=-1,
                                        random_state=seed) if kind == "cls"
                 else RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=seed))
        else:
            try: import xgboost as xgb
            except Exception: log(f"skip {tag}: xgboost not installed"); return
            common = dict(n_estimators=600, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=seed)
            m = (xgb.XGBClassifier(**common, num_class=(n_out if n_out > 2 else None),
                                   objective=("multi:softprob" if n_out > 2 else "binary:logistic"))
                 if kind == "cls" else xgb.XGBRegressor(**common))
        m.fit(Xs[i_tr], y[i_tr])
        pt = m.predict(Xs[i_te]); pe = m.predict(Xe)
        prob_e = m.predict_proba(Xe) if (kind == "cls" and hasattr(m, "predict_proba")) else None
        synth_val = None
    else:
        amp = _amp_dtype(dev); use_amp = dev.type == "cuda"
        # mlp: Xs is 2D (d_in = shape[1], possibly flattened seq); cnn1d/transformer1d:
        # Xs is 3D (C = shape[1], L = shape[2]). length only used by transformer1d.
        length = Xs.shape[2] if (is_seq and model != "mlp") else 1601
        net = _build_tab_model(model, feature, Xs.shape[1], n_out, kind,
                               length=length).to(dev)
        Xtr = torch.from_numpy(Xs).to(dev)
        yt_t = torch.from_numpy(y if kind == "cls" else y.astype(np.float32)).to(dev)
        cls_w = None
        if kind == "cls":
            cnt = np.bincount(y[i_tr], minlength=n_out).astype(np.float32)
            cls_w = torch.tensor(cnt.sum() / np.clip(cnt * n_out, 1e-6, None)).float().to(dev)
        lossf = nn.CrossEntropyLoss(weight=cls_w) if kind == "cls" else nn.MSELoss()
        need_scaler = use_amp and amp == torch.float16
        try: scaler = torch.amp.GradScaler("cuda", enabled=need_scaler)
        except (AttributeError, TypeError): scaler = torch.cuda.amp.GradScaler(enabled=need_scaler)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5, min_lr=1e-6)
        ck = out_dir / "models" / f"{tag}.ckpt"; bp = out_dir / "models" / f"{tag}.best.pt"
        start_ep = 0; best = -1e9; since = 0
        if ck.exists():
            c = torch.load(ck, map_location=dev); net.load_state_dict(c["model"])
            start_ep = c["epoch"]; best = c["best"]; since = c["since"]
            log(f"    {tag}: resume @epoch {start_ep}")

        def predict_idx(idx):
            net.eval(); out = []
            with torch.no_grad():
                for i in range(0, len(idx), 1024):
                    xb = Xtr[idx[i:i+1024]]
                    with torch.autocast("cuda", dtype=amp, enabled=use_amp):
                        o = net(xb).float().cpu().numpy()
                    out.append(o)
            return np.concatenate(out)

        for ep in range(start_ep, max_epochs):
            net.train(); perm = np.random.permutation(i_tr)
            for j in range(0, len(perm), batch):
                bi = perm[j:j+batch]; xb = Xtr[bi]; yb = yt_t[bi]
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=amp, enabled=use_amp):
                    o = net(xb); loss = lossf(o, yb.long()) if kind == "cls" else lossf(o, yb.float().unsqueeze(1))
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            ov = predict_idx(i_va)
            pv = ov.argmax(1) if kind == "cls" else ov.squeeze(1)
            from sklearn.metrics import f1_score
            m_ = (f1_score(y[i_va], pv, labels=list(range(n_out)), average="macro", zero_division=0)
                  if kind == "cls" else r2_score(y[i_va], pv))
            sched.step(m_)
            if m_ > best + min_delta:
                best = float(m_); since = 0
                torch.save({k: v.detach().clone() for k, v in net.state_dict().items()}, bp)
            else: since += 1
            if ((ep+1) % ckpt_every == 0) or since >= patience:
                torch.save({"model": {k: v.detach().clone() for k, v in net.state_dict().items()},
                            "epoch": ep+1, "best": best, "since": since}, ck)
            if (ep+1) % 10 == 0 or since >= patience:
                log(f"    {tag} ep{ep+1} val={m_:+.4f} best={best:+.4f} since={since} ({time.time()-t0:.0f}s)")
            if since >= patience: log(f"    {tag}: converged"); break
        if bp.exists(): net.load_state_dict(torch.load(bp, map_location=dev))
        # synth-test + exp preds
        ot = predict_idx(i_te); pt = ot.argmax(1) if kind == "cls" else ot.squeeze(1)
        net.eval(); pe_raw = []
        Xe_t = torch.from_numpy(Xe).to(dev)
        with torch.no_grad():
            for i in range(0, len(Xe_t), 1024):
                with torch.autocast("cuda", dtype=amp, enabled=use_amp):
                    pe_raw.append(net(Xe_t[i:i+1024]).float().cpu().numpy())
        oe = np.concatenate(pe_raw)
        if kind == "cls":
            pe = oe.argmax(1); ex = np.exp(oe - oe.max(1, keepdims=True)); prob_e = ex / ex.sum(1, keepdims=True)
        else: pe = oe.squeeze(1); prob_e = None
        synth_val = best
        for _p in (ck, bp):
            try: _p.unlink()
            except FileNotFoundError: pass

    # ---- metrics + per-case JSON ----
    if kind == "cls":
        s_acc, s_mf1, s_bal = _metrics_cls(y[i_te], pt.astype(int), n_out)
    else:
        s_acc = float(r2_score(y[i_te], pt)); s_mf1 = s_bal = None
    rows = []
    for i, ix in enumerate(idx_e):
        r = {"case": exp_names[ix],
             "y_true": (int(e_y[i]) if e_kind == "cls" else float(e_y[i])),
             "y_pred": (int(pe[i]) if e_kind == "cls" else float(pe[i]))}
        if e_kind == "cls" and prob_e is not None: r["proba"] = [float(p) for p in prob_e[i]]
        rows.append(r)
    meta = {"task": task, "model": model, "feature": feature, "n_target": res,
            "kind": e_kind, "n_out": int(n_out), "subsample": int(min(subsample, len(idx_pool))),
            "synth_val_best": synth_val, "synth_test_metric": float(s_acc),
            "synth_test_macro_f1": (float(s_mf1) if s_mf1 is not None else None),
            "synth_test_bal_acc": (float(s_bal) if s_bal is not None else None),
            "standardized": (not is_seq), "trained_to_convergence": model in NN_MODELS}
    pc.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    sp = out_dir / "synth_test_tab.json"
    allr = json.loads(sp.read_text()) if sp.exists() else {}
    allr[tag] = meta; sp.write_text(json.dumps(allr, indent=2))
    log(f"  DONE {tag}: synth {'mF1' if kind=='cls' else 'R2'}="
        f"{(s_mf1 if s_mf1 is not None else s_acc):.3f}")


def tab_cells(models: Sequence[str], tasks: Sequence[str]) -> List[Tuple[str, str, str]]:
    out = []
    for t in tasks:
        for m in models:
            for f in TAB_MODEL_FEATURES.get(m, []):
                out.append((t, m, f))
    return out
