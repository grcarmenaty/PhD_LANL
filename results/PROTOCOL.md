# Train / validation / test protocol

This document pins down what the words "train", "val", "test", and
"experimental evaluation" mean throughout this project, so that the
metric numbers in the rest of `results/` are unambiguous.

## The four data populations

| name                | size  | source            | role                                              |
|---------------------|-------|-------------------|---------------------------------------------------|
| **train**           | 7 000 | synthetic         | fit model parameters during HPO                   |
| **validation**      | 1 500 | synthetic         | HPO selects the best `(hyperparameter cell)` here |
| **synthetic test**  | 1 500 | synthetic         | reported final in-domain metric (held out from HPO) |
| **experimental**    |    61 | IQS lab `median_frfs.h5` | cross-domain transfer test (sim-to-real)   |

The 10 000-sample synthetic dataset is split 70 / 15 / 15 with
seed `20260511`.  The split is stratified on class for
classification tasks and randomised for regression.  See
[`../ml_pipeline/train.py`](../ml_pipeline/train.py) — function
`make_split` — for the exact implementation.

## Why two tests instead of one

Earlier feedback proposed "train + validate on synth, test on
experimental".  What I shipped is a strict superset of that:

* **The experimental cases are tested** exactly as proposed —
  `evaluate.py` runs every HPO-selected model on all 61 IQS cases
  and reports accuracy / R² in
  [`experimental_evaluation.json`](experimental_evaluation.json).
* **In addition**, a 15 % slice of the synthetic data is held out
  from HPO and only scored once after HPO ends.  That extra slice
  is the "synth test" / "test" column in [`RESULTS.md`](RESULTS.md)
  and in the per-cell HPO files in [`hpo/`](hpo/).

Keeping both serves two different purposes:

1. **In-domain generalisation** — a single number per `(task,
   model, feature)` cell that captures how well the model
   generalises to *unseen* synthetic samples from the same
   distribution it trained on.  Without this, a model that
   overfits the train fold can still look fine on a noisy
   sim-to-real evaluation.
2. **Cross-domain transfer** — a separate number per cell on the
   IQS experimental data, which has real noise, calibration
   tolerances, and composite damage scenarios.  This is the
   "sim-to-real gap" indicator.

A drop from synth test to experimental tells you the magnitude of
the domain shift; without the synth test as anchor you cannot tell
whether a poor experimental score is a sim-to-real problem or a
fundamental modelling failure.

## What is *not* used for HPO

The validation fold is the **only** signal used to pick
hyperparameters.  The synth test fold and the experimental cases
are never seen during training or model selection — they are only
scored once at the end.  This protects the headline metric from
HPO leakage.

## How to reproduce the split

```python
from ml_pipeline.train import load_labels, make_split
from ml_pipeline.tasks import build_targets

labels = load_labels("dataset/features.h5")
tasks  = build_targets(labels["type_code"], labels["storey"],
                         labels["end"], labels["severity"])
for task_name, (mask, y_pool, kind) in tasks.items():
    idx_train, idx_val, idx_test = make_split(y_pool, kind)
    # ...
```

## Headline metric definitions

* Classification tasks (`binary`, `type`, `col_location`,
  `mass_location`): `accuracy = correct / total` on the test fold.
  Per-class breakdowns (recall, F1) are in
  [`PLOTS.md`](PLOTS.md) and the per-task documents under
  [`by_task/`](by_task/).
* Regression task (`severity`): coefficient of determination
  `R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)²`.  Reported alongside the mean
  absolute error of the predicted normalised severity.

The class baseline (random or class-prior) is included next to every
headline number so the reader can immediately see whether a model
is doing better than chance.
