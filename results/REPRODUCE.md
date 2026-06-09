# Reproducing every figure in the consolidated / synth reports

Two resolution studies, each with a consolidated + in-domain report, all
regenerable **from data committed in this repo** with one command per script:

| resolution | consolidated | in-domain companion |
|---|---|---|
| 1601-bin (native)    | `REPORT_CONSOLIDATED.md`     | `REPORT_synth.md`     |
| 128-bin  (decimated) | `REPORT_CONSOLIDATED_128.md` | `REPORT_synth_128.md` |

Every script takes `--res {1601,128}` (default 1601). 1601 figures live in
`results/figures/hires/`, 128 figures in `results/figures/hires128/`; 128 data
artefacts carry a `_128` suffix (`analysis_128.json`, …; DT uses `dt_{res}.json`).

## TL;DR — regenerate both report sets

```bash
# from the repo root; run for RES=1601 then RES=128
for RES in 1601 128; do
  python ml_pipeline/hires_zoo_summary.py            # per-cell distillation (both res) -> zoo_summary.json
  python ml_pipeline/hires_dt_1601.py        --res $RES   # DT balanced-acc sweep + is_bolt fig
  python ml_pipeline/hires_dt_diag.py        --res $RES   # DT-swept AUC/confusion (+2 figs)
  python ml_pipeline/hires_dt_stiffness.py   --res $RES   # DT vs stiffness loss (+2 figs)
  python ml_pipeline/hires_arch.py           --res $RES   # measured model sizes (+fig)
  python ml_pipeline/hires_compute.py        --res $RES   # FLOPs / training effort (+fig)
  python ml_pipeline/hires_analysis.py       --res $RES   # EDA + best-cell diagnostics (+6 figs)
  python ml_pipeline/hires_inputs.py         --res $RES   # input-sample figures (+4 figs)
  python ml_pipeline/build_hires_synth_report.py --res $RES   # in-domain companion report
  python ml_pipeline/build_hires_report.py   --res $RES   # consolidated report + Figure 1 + cellzoo_*
done
```

`hires_zoo_summary.py` distils ALL resolutions at once (it only needs running
once). No GPU, no Google Drive, no `/tmp` state required: the scripts read the
committed artefacts below and auto-extract the prediction archives on first use.

## Committed data the figures consume

| artefact | what it is | built by |
|---|---|---|
| `results_hires/per_case_hires1601.tar.gz` | 575 unique **1601-bin per-case predictions** (`y_true`, `y_pred`, class probabilities) | `build_figure_bundle.py --res 1601` |
| `results_hires/per_case_hires128.tar.gz` | 570 unique **128-bin per-case predictions** | `build_figure_bundle.py --res 128` |
| `results_hires/figure_data.npz` / `figure_data_128.npz` | compact cache of the FRF-derived arrays the EDA/input figures need, at each resolution (channel-mean log\|FRF\|, labels, sample complex FRFs) | `build_figure_bundle.py --res {1601,128}` |
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
