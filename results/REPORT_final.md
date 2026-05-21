# REPORT_final.md — retired

This standalone report has been **retired**. The reporting set is now
organised **by diagnosis goal** (damage detection, type, severity,
location), which made this document redundant.

Its content lives in:

* [`REPORT_definitive.md`](REPORT_definitive.md) — concise, goal-structured,
  with the exact training conditions per goal.
* [`REPORT_full.md`](REPORT_full.md) — the detailed companion. The joint
  synth+exp fine-tune that this report covered (the only approach shown to
  reach deployable accuracy, since it uses experimental labels in training)
  is in **`REPORT_full.md` § 9.2**; the per-task synth-only results are in
  `REPORT_full.md` §§ 5–8.

Per-case prediction artefacts for the joint fine-tune (5 seeds, best-by-
metric kept) remain in `results/per_case_final/`. The prior revision of this
file is in git history.
