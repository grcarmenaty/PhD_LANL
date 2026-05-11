"""Task / target definitions for the 5 ML problems.

  binary       : Pristine vs damage          (2-class)
  type         : 5-class damage classification (Pristine, Bolt, Crack, Hole, Mass)
  severity     : severity regression          (damage samples only; per-type normalised)
  col_location : column damage location       (Bolt/Crack/Hole only; 6 classes = 3 storey x 2 end)
  mass_location: which plate carries the mass (Mass only; 4 classes)
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ml_pipeline.case_design import (
    TYPE_PRISTINE, TYPE_BOLT, TYPE_CRACK, TYPE_HOLE, TYPE_MASS,
    SEVERITY_BOUNDS, END_BD, END_AD,
)


# ── helper masks ────────────────────────────────────────────────────────────
def is_damage(type_code: np.ndarray) -> np.ndarray:
    return type_code != TYPE_PRISTINE


def is_column_damage(type_code: np.ndarray) -> np.ndarray:
    return (type_code == TYPE_BOLT) | (type_code == TYPE_CRACK) | (type_code == TYPE_HOLE)


def is_mass(type_code: np.ndarray) -> np.ndarray:
    return type_code == TYPE_MASS


# ── per-task (mask, target) builders ────────────────────────────────────────
def build_targets(type_code: np.ndarray, storey: np.ndarray,
                   end: np.ndarray, severity: np.ndarray
                   ) -> Dict[str, Tuple[np.ndarray, np.ndarray, str]]:
    """Return ``{task_name: (mask, y, kind)}`` where ``kind`` is 'cls' or 'reg'.

    The mask selects which dataset rows participate in that task; ``y`` is
    the target restricted to those rows.
    """
    tasks: Dict[str, Tuple[np.ndarray, np.ndarray, str]] = {}

    # ---- 1. binary ----------------------------------------------------------
    mask = np.ones_like(type_code, dtype=bool)
    y = is_damage(type_code).astype(np.int64)
    tasks["binary"] = (mask, y, "cls")

    # ---- 2. damage type (5-class) ------------------------------------------
    mask = np.ones_like(type_code, dtype=bool)
    y = type_code.astype(np.int64)
    tasks["type"] = (mask, y, "cls")

    # ---- 3. severity regression --------------------------------------------
    mask = is_damage(type_code)
    sel  = type_code[mask]
    sev  = severity[mask].astype(np.float32)
    # Normalise to [0, 1] per damage type.
    y = np.zeros_like(sev, dtype=np.float32)
    for tc in (TYPE_BOLT, TYPE_CRACK, TYPE_HOLE, TYPE_MASS):
        lo, hi = SEVERITY_BOUNDS[tc]
        rows = sel == tc
        y[rows] = (sev[rows] - lo) / (hi - lo)
    tasks["severity"] = (mask, y, "reg")

    # ---- 4. column-damage location (6 classes) -----------------------------
    mask = is_column_damage(type_code)
    sty  = storey[mask].astype(np.int64)
    end2 = end[mask].astype(np.int64)
    # storey ∈ {0,1,2}, end ∈ {0(BD),1(AD)}  →  storey*2 + end
    y = (sty * 2 + end2).astype(np.int64)
    tasks["col_location"] = (mask, y, "cls")

    # ---- 5. mass plate location (4 classes) --------------------------------
    mask = is_mass(type_code)
    y = end[mask].astype(np.int64)         # plate idx stored in 'end' field
    tasks["mass_location"] = (mask, y, "cls")

    return tasks


# Class counts for display.
TASK_N_CLASSES = {
    "binary":        2,
    "type":          5,
    "severity":      None,   # regression
    "col_location":  6,
    "mass_location": 4,
}

TASK_DESCRIPTION = {
    "binary":        "Pristine vs Damage",
    "type":          "Damage type (Pristine/Bolt/Crack/Hole/Mass)",
    "severity":      "Severity regression (normalised [0,1] per type)",
    "col_location":  "Column-damage location (storey x end)",
    "mass_location": "Mass plate location (Base/F1/F2/F3)",
}
