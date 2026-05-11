"""Emit a Markdown dump of every HPO trial for every (task,
model, feature) cell.

The output is written to ``results/ALL_TRIALS.md`` and embedded /
linked from ``REPORT.md``.  Each cell becomes one table; every
row is one trial with its hyperparameter combination, the val
and test metric, and the runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _format_trials(blob: dict) -> list[str]:
    trials = blob.get("trials", [])
    if not trials:
        return []
    keys = sorted(trials[0]["hyperparams"].keys())
    head_cells = keys + ["val", "test", "runtime_s"]
    lines = ["| " + " | ".join(head_cells) + " |",
              "|" + "|".join(["---"] * len(head_cells)) + "|"]
    for t in trials:
        h = t["hyperparams"]
        vals = [str(h[k]) for k in keys]
        vals.append(f'{t["metric_val"]:.4f}'
                       if t["metric_val"] is not None else "—")
        vals.append(f'{t["metric_test"]:.4f}'
                       if t["metric_test"] is not None else "—")
        vals.append(f'{t.get("runtime_s", 0):.1f}')
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=_REPO / "results")
    args = p.parse_args()

    out = args.results / "ALL_TRIALS.md"
    blocks: list[str] = ["# Full HPO trial dump\n",
                            "Every trial for every `(task, model, feature)` cell."
                            "  Rows are sorted by `metric_val` "
                            "(descending).  See § 7 of "
                            "[`REPORT.md`](REPORT.md) for narrative context.\n"]

    # Classification / severity HPO
    blocks.append("## Classification + severity HPO\n")
    for path in sorted((args.results / "hpo").glob("*.json")):
        blob = json.loads(path.read_text())
        title = (f"### `{blob['task']}` / `{blob['model']}` / "
                    f"`{blob['feature']}`")
        n_trials = len(blob.get("trials", []))
        blocks.append(title)
        blocks.append(f"_{n_trials} trials_  ·  best val "
                          f"= **{blob.get('best_metric_val', None):.4f}**  ·  "
                          f"best test = **{blob.get('best_metric_test', None):.4f}**\n"
                          if blob.get("best_metric_val") is not None
                          else f"_{n_trials} trials_\n")
        # Sort trials by val desc
        trials = sorted(blob.get("trials", []),
                            key=lambda t: -(t.get("metric_val") or -1e30))
        sorted_blob = dict(blob); sorted_blob["trials"] = trials
        blocks.extend(_format_trials(sorted_blob))
        blocks.append("")

    # Indicator-prediction HPO
    blocks.append("\n## Indicator-prediction HPO\n")
    for path in sorted((args.results / "hpo_indicators").glob("*.json")):
        blob = json.loads(path.read_text())
        title = (f"### `{blob['indicator']}` / `{blob['model']}` / "
                    f"`{blob['feature']}`")
        n_trials = len(blob.get("trials", []))
        blocks.append(title)
        blocks.append(f"_{n_trials} trials_  ·  best val R² "
                          f"= **{blob.get('best_metric_val', None):.4f}**  ·  "
                          f"best test R² = **{blob.get('best_metric_test', None):.4f}**\n"
                          if blob.get("best_metric_val") is not None
                          else f"_{n_trials} trials_\n")
        trials = sorted(blob.get("trials", []),
                            key=lambda t: -(t.get("metric_val") or -1e30))
        sorted_blob = dict(blob); sorted_blob["trials"] = trials
        blocks.extend(_format_trials(sorted_blob))
        blocks.append("")

    out.write_text("\n".join(blocks))
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
