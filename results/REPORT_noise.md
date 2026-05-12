# Noisy-synth study — companion to `REPORT.md`

Every experiment in [`REPORT.md`](REPORT.md) is repeated on synthetic data corrupted by additive Gaussian noise on the **time-series** field (1024 × 9 acceleration samples per signal).  The noise is applied **per sample, per channel, at a controlled signal-to-noise ratio**; every downstream feature (FRF, modal-peak vector, CFDAC variants, pymodal indicators) is then re-extracted from the noisy time series so the entire pipeline trains and tests on a self-consistent noisy dataset.

Five SNR levels are evaluated: **35, 25, 20, 15, 10 dB**.  All other settings (model menu, HPO grid, balanced experimental evaluation, transfer-learning sweep, resolution sweep) are identical to the clean study.

Coverage status at the time of this build:

| SNR (dB) | features.h5 | HPO | indicator | balanced eval | transfer | resolution |
|---|---|---|---|---|---|---|
| **35**  | — | — | — | — | — | — |
| **25**  | — | — | — | — | — | — |
| **20**  | — | — | — | — | — | — |
| **15**  | — | — | — | — | — | — |
| **10**  | — | — | — | — | — | — |

_No per-SNR HPO results exist yet.  Run `python ml_pipeline/run_noise_sweep.py` to populate them._