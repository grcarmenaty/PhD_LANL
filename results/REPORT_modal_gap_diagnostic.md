# Modal-feature synth→real gap diagnostic

**Date:** 2026-05-28. **Trigger:** the v2a rejection
([`REPORT_v2a_chunk_regen.md`](REPORT_v2a_chunk_regen.md)) flagged the modal
feature pathway — not the damage geometry — as the suspected cause of the
col_location / Crack / Hole synth-real gap. This is a training-free probe
to locate the gap before committing more compute.

**Script:** [`ml_pipeline/diagnose_modal_gap.py`](../ml_pipeline/diagnose_modal_gap.py)
(logistic-regression transfer probes on the 81-dim `modal` feature;
no pipeline training, ~30 s).

The 81-dim `modal` feature = 9 channels × {peak1/2/3 freq, peak1/2/3
log-amp, mean log-amp, std log-amp, band energy}.

## Finding 1 — the feature is dominated by non-transferable absolute magnitude

Per-dim covariate shift (experimental mean on the synth z-scale), averaged
within each of the 9 channel stats:

| stat | mean \|shift\| (σ) |
|---|---|
| pk1_f, pk2_f, pk3_f (peak **frequencies**) | 0.2 – 0.4 |
| pk1_a, pk2_a, pk3_a (peak **log-amplitudes**) | 6 – 8 |
| mean_logA (absolute level) | 8.9 |
| std_logA | 4.8 |
| **band_E (Σ amp²)** | **~9 × 10⁹** |

47 of 81 dims shift more than 2σ. The **frequencies transfer well**; every
**absolute-amplitude / energy** stat is shifted by many σ (band energy
absurdly so — it is an un-normalised absolute quantity that depends entirely
on excitation level and sensor gain, which differ between the synth
generator and the IQS rig).

## Finding 2 — the synth discriminant *inverts* on real data

Logistic regression trained on synth z-features, evaluated in-domain
(5-fold CV) vs transferred to the experimental bookcase:

| task | synth CV BA | exp transfer BA | mean-diff cosine(syn,exp) |
|---|---|---|---|
| is_hole | 0.744 | **0.400** (below chance) | **−0.296** (inverted) |
| is_crack | 0.683 | 0.496 | **−0.321** (inverted) |

A negative cosine means the direction that separates the classes in synth
points the **opposite way** on real data — the synth-trained model is
actively misled, not merely uninformative. The `band_E` dimension's
class-mean-difference sign agrees between domains in **0 %** of channels.

## Finding 3 — gain-invariant features remove the inversion

Rebuilding the feature to be gain-invariant (keep peak frequencies + std
log-amp; replace absolute peak/mean log-amps with channel-mean-centred
amplitudes; drop band energy → 63 dims):

| task | feature | synth CV | exp BA | cosine |
|---|---|---|---|---|
| is_hole | full-81 | 0.744 | 0.400 | −0.296 |
| is_hole | **gain-inv-63** | 0.678 | **0.497** | **+0.552** |
| is_crack | full-81 | 0.683 | 0.496 | −0.321 |
| is_crack | gain-inv-63 | 0.656 | 0.489 | −0.004 |

De-magnituding flips the is_hole discriminant from inverted (−0.30) to
aligned (+0.55) and pulls transfer from below-chance up to chance; for
is_crack it removes the inversion (−0.32 → ~0). The small drop in synth
in-domain CV is exactly the non-transferable absolute-magnitude signal being
discarded. **Gain-invariance stops the model being actively wrong, but does
not by itself add transferable separation** — the remaining gap is genuine.

## Finding 4 — col_location is a *generation* problem, not a transfer problem

Among crack/hole cases, classifying the damaged floor-end (BD vs AD) from
synth modal features scores **0.497 balanced-acc in-domain (chance)**. v1's
symmetric damage encodes no column/end side information in the modal
feature, so col_location is unlearnable from v1 synth before transfer even
enters the picture. This is why v2/v2a tried asymmetric damage — the instinct
was right — but Finding 1–2 explain why it still failed on real data: the
weak column signal the asymmetric geometry created was swamped by the same
non-transferable absolute-magnitude covariate shift.

## Recommendations (cheap → expensive)

1. **Make the modal feature gain/level-invariant** (drop band energy and
   absolute log-amps; keep frequencies, amplitude *ratios*, std log-amp; or
   per-sample-normalise |H(f)| before extraction). Cheap: rebuild the
   feature from existing FRFs, no regeneration. Expected to stop the
   is_hole/is_crack discriminant inversion (validated above). This alone
   won't lift transfer above chance, but it removes an active failure mode
   and is a prerequisite for anything else.
2. **For col_location specifically**, asymmetric damage *and* gain-invariant
   features are both necessary — neither alone suffices. Re-test the v2a
   asymmetric geometry on a gain-invariant modal feature before discarding
   the idea entirely.
3. Only if 1–2 show in-domain↔real alignment is worth a full re-train should
   another 3-seed sweep be launched.
