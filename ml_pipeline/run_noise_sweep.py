"""Run the full pipeline (features → HPO → indicator regression →
transfer learning → resolution sweep) for one or more noisy SNR
variants of the synth dataset.

For each SNR ∈ ``SNR_LEVELS_DB`` this script:

  1. Re-uses ``build_noisy_chunks.py`` to drop noisy chunks under
     ``dataset/noisy_<SNR>dB/``.
  2. Runs ``features.py`` to extract every base feature into
     ``dataset/features_noisy_<SNR>dB.h5``.
  3. Runs ``cfdac.py`` / ``cfdac_variants.py`` to add the 10 CFDAC
     variant arrays.
  4. Runs ``hpo.py`` and ``hpo_cfdac_variants.py`` + ``hpo_cfdac_allmodels.py``
     against the noisy features file, writing results to
     ``results/noisy_<SNR>dB/hpo/``.
  5. Runs ``train_indicator_predictors.py`` against the noisy file.
  6. Re-runs the balanced experimental eval, transfer-learning sweep,
     and resolution sweep using the noisy synth-trained models.

Each step is resume-friendly: if its target output exists the step is
skipped.  Stop / restart at any time and re-run.

Outputs (per SNR):
    dataset/noisy_<SNR>dB/chunk_*.h5
    dataset/features_noisy_<SNR>dB.h5
    results/noisy_<SNR>dB/
        ├── hpo/*.json
        ├── models/*.{pt,pkl}
        ├── hpo_indicators/*.json
        ├── balanced/experimental_full_evaluation.json
        ├── transfer_learning.json
        └── resolution_sweep.json
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

_REPO = Path(__file__).resolve().parent.parent

SNR_LEVELS_DB = (35.0, 25.0, 20.0, 15.0, 10.0)


def _run(cmd: List[str], log: Path) -> bool:
    print(f"+ {' '.join(map(str, cmd))}", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "ab") as fh:
        fh.write(f"\n--- $ {' '.join(map(str, cmd))}\n".encode())
        result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    ok = result.returncode == 0
    print(f"  → exit {result.returncode}  ({dt:.1f}s)  log: {log}",
              flush=True)
    return ok


def _exists(p: Path) -> bool:
    return p.exists()


def run_snr(snr_db: float, log_root: Path,
                skip_resolution: bool = False) -> None:
    snr_int = int(round(snr_db))
    log_dir = log_root / f"noisy_{snr_int}dB"
    log_dir.mkdir(parents=True, exist_ok=True)

    chunk_dir   = _REPO / "dataset" / f"noisy_{snr_int}dB"
    feat_file   = _REPO / "dataset" / f"features_noisy_{snr_int}dB.h5"
    results_dir = _REPO / "results" / f"noisy_{snr_int}dB"

    # 1. noisy chunks
    if not chunk_dir.exists() or not list(chunk_dir.glob("chunk_*.h5")):
        _run([sys.executable, "ml_pipeline/build_noisy_chunks.py",
                  "--snr-db", str(snr_db)], log_dir / "build_chunks.log")

    # 2. base features
    if not feat_file.exists():
        _run([sys.executable, "ml_pipeline/features.py",
                  "--dataset", str(chunk_dir),
                  "--out", str(feat_file)], log_dir / "features.log")

    # 3. CFDAC + variants (in-place additions to feat_file)
    _run([sys.executable, "ml_pipeline/cfdac.py",
              "--features", str(feat_file)], log_dir / "cfdac.log")
    _run([sys.executable, "ml_pipeline/cfdac_variants.py",
              "--features", str(feat_file)], log_dir / "cfdac_variants.log")

    # 4. HPO sweeps (full grid + cfdac variants + all-models)
    results_dir.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "ml_pipeline/hpo.py",
              "--features", str(feat_file),
              "--out", str(results_dir)], log_dir / "hpo.log")
    _run([sys.executable, "ml_pipeline/hpo_cfdac_variants.py",
              "--features", str(feat_file),
              "--out", str(results_dir)], log_dir / "hpo_cfdac_variants.log")
    _run([sys.executable, "ml_pipeline/hpo_cfdac_allmodels.py",
              "--features", str(feat_file),
              "--out", str(results_dir)], log_dir / "hpo_cfdac_allmodels.log")

    # 5. indicator regression
    _run([sys.executable, "ml_pipeline/train_indicator_predictors.py",
              "--features", str(feat_file),
              "--out", str(results_dir)], log_dir / "indicators.log")

    # 6. balanced experimental evaluation
    balanced_exp = _REPO / "dataset" / "experimental_features_balanced.h5"
    bal_out = results_dir / "balanced"
    bal_out.mkdir(parents=True, exist_ok=True)
    # Symlink the model dirs so evaluate_full_experimental.py can find them
    for sub in ("models", "models_indicators"):
        link = bal_out / sub
        if not link.exists():
            try:
                link.symlink_to(results_dir / sub)
            except FileExistsError:
                pass
    _run([sys.executable, "ml_pipeline/evaluate_full_experimental.py",
              "--syn", str(feat_file),
              "--exp", str(balanced_exp),
              "--out", str(bal_out)], log_dir / "evaluate.log")

    # 7. transfer learning sweep (uses balanced exp)
    _run([sys.executable, "ml_pipeline/transfer_learn.py",
              "--syn", str(feat_file),
              "--exp", str(balanced_exp),
              "--results", str(results_dir),
              "--epochs", "4"], log_dir / "transfer.log")

    # 8. resolution sweep (synth-only — does not touch experimental)
    if not skip_resolution:
        _run([sys.executable, "ml_pipeline/resolution_sweep.py",
                  "--features", str(feat_file),
                  "--out", str(results_dir),
                  "--epochs", "3"], log_dir / "resolution.log")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snr-db", type=float, nargs="*",
                      default=list(SNR_LEVELS_DB),
                      help="One or more SNR levels (dB) to process.")
    p.add_argument("--log-root", type=Path,
                      default=_REPO / "results" / "noise_sweep_logs")
    p.add_argument("--skip-resolution", action="store_true",
                      help="Skip the per-SNR resolution sweep "
                              "(saves ~80 min per SNR).")
    args = p.parse_args()

    for snr in args.snr_db:
        print(f"\n=================  SNR = {snr:.1f} dB  =================\n",
                  flush=True)
        run_snr(snr, args.log_root, skip_resolution=args.skip_resolution)
    print("\ndone.")


if __name__ == "__main__":
    main()
