# Per-task documentation

Self-contained walkthrough of every `(model, feature)` cell on
each of the five ML tasks.  Each task document **repeats** the
model and feature explanations so you do not need to jump back
to a central glossary.

| task | classes | best cell | synth test | exp test |
|------|---------|-----------|------------|----------|
| [01 — Binary: Pristine vs Damage](01_binary.md) | 2 | MLP + modal | 0.989 | 0.869 (class baseline) |
| [02 — Type: 5-class damage type](02_type.md)   | 5 | MLP + modal | 0.877 | 0.443 |
| [03 — Severity regression](03_severity.md)     | — | RF + modal  | R² 0.573 | R² −0.15 |
| [04 — Column damage location](04_col_location.md) | 6 | MLP + modal | 0.494 | 0.490 |
| [05 — Mass plate location](05_mass_location.md) | 4 | RF + modal  | 0.990 | 0.250 (n=4) |

If you only have time for one document, read
[`01_binary.md`](01_binary.md) — it has the most thoroughly
worked-through example of every plot type.

The shared introduction (model definitions, feature
definitions, synth-vs-real example panels) lives at
[`_template_intro.md`](_template_intro.md) and is duplicated
into the head of every task document.

Cross-references:

* Train / val / test protocol — [`../PROTOCOL.md`](../PROTOCOL.md).
* How to read every plot type — [`../INTERPRETING_PLOTS.md`](../INTERPRETING_PLOTS.md).
* Per-plot commentary (all 146 plots) — [`../PLOTS.md`](../PLOTS.md).
* Executive summary — [`../RESULTS.md`](../RESULTS.md).
