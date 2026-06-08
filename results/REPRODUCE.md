# Reproducing every figure in `REPORT_CONSOLIDATED.md` / `REPORT_synth.md`

Everything the two reports show can be regenerated **from data committed in this
repo**, with one command per script. This file documents the chain and what each
artefact depends on.

## TL;DR — regenerate all figures + the report

```bash
# from the repo root
python ml_pipeline/hires_zoo_summary.py      # per-cell distillation  -> zoo_summary.json (+2 figs)
python ml_pipeline/hires_dt_1601.py          # DT balanced-acc sweep   -> dt_1601.json (+fig)
python ml_pipeline/hires_dt_diag.py          # DT-swept AUC/confusion  -> dt_diag.json (+2 figs)
python ml_pipeline/hires_dt_stiffness.py     # DT vs stiffness loss    -> dt_stiffness.json (+2 figs)
python ml_pipeline/hires_analysis.py         # EDA + best-cell diagnostics (+6 figs)
python ml_pipeline/hires_arch.py             # measured model sizes    -> architectures.json (+fig)
python ml_pipeline/hires_inputs.py           # input-sample figures (+4 figs)
python ml_pipeline/build_hires_report.py     # assembles the report + Figure 1 + cellzoo_* figs
```

No GPU, no Google Drive, no `/tmp` state required. The scripts read the two
committed artefacts below and auto-extract the prediction archive on first use.

## Committed data the figures consume

| artefact | what it is | built by |
|---|---|---|
| `results_hires/per_case_hires1601.tar.gz` | the 575 unique 1601-bin **per-case predictions** (`y_true`, `y_pred`, class probabilities) — the GPU training outputs | `build_figure_bundle.py` |
| `results_hires/figure_data.npz` | compact cache of the FRF-derived arrays the EDA/input figures need (channel-mean log\|FRF\| for synth-4000 + all exp, full labels, and the complex FRFs of the sample cases) | `build_figure_bundle.py` |
| `results_hires/*.json` | distilled per-cell metrics, DT sweeps, analysis stats | the analysis scripts above |

`ml_pipeline/figdata.py` is the single access layer: it extracts the archive to
`results_hires/per_case_all/` (git-ignored) on first use and serves labels /
EDA arrays / sample FRFs from the bundle, **falling back to the raw hi-res HDF5
and `/tmp/allres` automatically** if a developer has them.

## Where the data originally comes from (full provenance)

1. **Per-case predictions** — produced by training the model zoo on GPU
   (`notebooks/hires_*_gpu.ipynb`, engines `ml_pipeline/hires_{zoo,tab,all}.py`);
   raw outputs live on the `colab-hires-{tabular,cnn,transformer,vision}` branches.
   These are the **only artefact not cheaply regenerable on CPU**, which is why
   they are archived into the repo. Rebuild the archive with:
   ```bash
   python ml_pipeline/build_figure_bundle.py --root <dir with per_case/ trees>
   ```

2. **Hi-res FRF HDF5** (`dataset/features_hires.h5`, `dataset/experimental_features_hires.h5`)
   — git-ignored (≈1.2 GB) but fully regenerable from committed raw data, and
   only needed if you rebuild `figure_data.npz` from scratch:
   - synth: `python ml_pipeline/generate_dataset.py --n-t 4096` (the reduced-order
     simulator — no external input) → `dataset_hires/chunk_*.h5` →
     `python ml_pipeline/build_hires_synth_features.py`
   - exp: reassemble `experimental_frfs_chunks/*.part_*` → `experimental_frfs.h5` →
     `python ml_pipeline/build_hires_exp_features.py`

## Verifying self-sufficiency

The pipeline was checked by moving the hi-res HDF5 and `/tmp/allres` out of the
way and deleting `results_hires/per_case_all/`: all 28 report figures regenerate
identically from the committed archive + bundle alone (domain-classifier
AUC = 1.000, identical per-task metrics).
