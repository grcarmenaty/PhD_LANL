"""Prepend a deprecation banner to every legacy REPORT_*.md.
Idempotent — skips files that already have the banner.
"""
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
RESULTS = _REPO / "results"

BANNER = """> **DEPRECATED — consolidated.** This report has been superseded by
> [`REPORT_CONSOLIDATED.md`](REPORT_CONSOLIDATED.md), the single source of
> truth covering all 244 cells × 3 variants × 3 seeds with embedded plots
> and a cell-by-cell review. Content below is kept for historical reference.

---

"""

LEGACY = [
    "REPORT.md",
    "REPORT_full.md",
    "REPORT_definitive.md",
    "REPORT_final.md",
    "REPORT_dt_sweep.md",
    "REPORT_dt_3way_verdict.md",
    "REPORT_dt_3way_full.md",
    "REPORT_v2_chunk_regen.md",
    "REPORT_v2a_chunk_regen.md",
    "REPORT_severity_stratified.md",
    "REPORT_simtoreal.md",
    "REPORT_noisy_mixed.md",
    "REPORT_modal_gap_diagnostic.md",
    "REPORT_noise.md",
    "REPORT_vision.md",
    "REPORT_vision_v2.md",
]

def main():
    for name in LEGACY:
        p = RESULTS / name
        if not p.exists():
            print(f"  skip (missing): {name}")
            continue
        text = p.read_text()
        if "DEPRECATED — consolidated" in text:
            print(f"  skip (already deprecated): {name}")
            continue
        p.write_text(BANNER + text)
        print(f"  deprecated: {name}")

if __name__ == "__main__":
    main()
