"""Generate the synthetic time-series dataset with PRISTINE-anchored damage.

Identical to ``generate_dataset.py`` in every respect (sampling schedule,
deterministic chirp, chunking, labels, CLI flags) except that the
parameter→geometry map is swapped for ``pristine_physics``'s first-principles,
pristine-anchored submodels — so the damage magnitudes carry **no information
from the damaged experimental FRFs**.  The default output directory is
``dataset_pristine/`` so the calibrated dataset is never overwritten.

Usage (mirrors generate_dataset.py)::

    python ml_pipeline/generate_dataset_pristine.py \
        --out dataset_pristine --n-t 4096 --fs 256

``generate_dataset.simulate_sample`` resolves ``geometry_from_params`` from its
own module globals at call time, so reassigning that name here cleanly swaps the
damage model without touching the calibrated source.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import ml_pipeline.generate_dataset as gd                       # noqa: E402
from ml_pipeline.pristine_physics import geometry_from_params_pristine  # noqa: E402

# Swap the damage model: physics submodels anchored only on the pristine case.
gd.geometry_from_params = geometry_from_params_pristine


def main() -> None:
    if "--out" not in sys.argv:
        sys.argv += ["--out", str(_REPO / "dataset_pristine")]
    gd.main()


if __name__ == "__main__":
    main()
