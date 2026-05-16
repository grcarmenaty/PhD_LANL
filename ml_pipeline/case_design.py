"""Case-design enumeration for the 10000-sample synthetic dataset.

Five coarse damage types, each represented at every physical location
the LANL 3SBB benchmark admits, with bounded continuous severity.

Layout
------
Pristine : 1 class,  2000 samples           (no location, no severity)
Bolt     : 6 locs    (3 storeys x 2 ends),  2000 samples total
Crack    : 6 locs    (3 storeys x 2 ends),  2000 samples total
Hole     : 6 locs    (3 storeys x 2 ends),  2000 samples total
Mass     : 4 locs    (base + 3 floors),     2000 samples total
                                            ─────
                                            10000 samples

The 2000 samples within each non-pristine type are split as evenly as
possible across that type's locations; the residue is spread to the
first few locations to keep the per-type count exact.

Within each (type, location) bucket the severity value is drawn
uniformly at random within the physical bounds below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


# Damage-type integer codes (used by all ML labels).
TYPE_PRISTINE = 0
TYPE_BOLT     = 1
TYPE_CRACK    = 2
TYPE_HOLE     = 3
TYPE_MASS     = 4

TYPE_NAMES = {
    TYPE_PRISTINE: "Pristine",
    TYPE_BOLT:     "Bolt",
    TYPE_CRACK:    "Crack",
    TYPE_HOLE:     "Hole",
    TYPE_MASS:     "Mass",
}

# Severity bounds per damage type (units shown in comments).
SEVERITY_BOUNDS = {
    TYPE_PRISTINE: (0.0,  0.0),   # n/a
    TYPE_BOLT:     (5.0, 95.0),   # percent loosening
    TYPE_CRACK:    (1.0,  8.0),   # mm
    TYPE_HOLE:     (1.0,  6.0),   # mm
    TYPE_MASS:     (0.1,  2.5),   # kg
}

# Severity units (for downstream readers).
SEVERITY_UNITS = {
    TYPE_PRISTINE: "",
    TYPE_BOLT:     "percent",
    TYPE_CRACK:    "mm",
    TYPE_HOLE:     "mm",
    TYPE_MASS:     "kg",
}

# End codes for bolt/crack/hole: BD = bottom of column, AD = top.
END_BD = 0
END_AD = 1
END_NA = -1

# Mass plate codes 0..3 (Base, Floor 1, Floor 2, Floor 3).  Stored in the
# `end` field for mass cases so the same label tuple shape applies to all
# damage types.
MASS_PLATE_NAMES = {0: "Base", 1: "First Floor",
                    2: "Second Floor", 3: "Third Floor"}


@dataclass
class CaseLocation:
    """One physical realisation of (damage type, location)."""

    type_code: int
    storey: int        # -1 if N/A (Pristine, or Mass uses plate in `end`)
    end: int           # END_BD, END_AD, or -1
    label: str

    def __str__(self) -> str:
        return self.label


def _bolt_label(storey: int, end: int) -> str:
    return f"D 1AD storey{storey+1}" if end == END_AD else f"D 1BD storey{storey+1}"


def enumerate_locations() -> List[CaseLocation]:
    """Return all 23 (type, location) buckets used to stratify the dataset."""
    out: List[CaseLocation] = []
    out.append(CaseLocation(TYPE_PRISTINE, -1, -1, "Pristine"))
    for s in (0, 1, 2):
        for e, etag in ((END_BD, "BD"), (END_AD, "AD")):
            out.append(CaseLocation(TYPE_BOLT,  s, e, f"Bolt {s+1}{etag}"))
    for s in (0, 1, 2):
        for e, etag in ((END_BD, "BD"), (END_AD, "AD")):
            out.append(CaseLocation(TYPE_CRACK, s, e, f"Crack {s+1}{etag}"))
    for s in (0, 1, 2):
        for e, etag in ((END_BD, "BD"), (END_AD, "AD")):
            out.append(CaseLocation(TYPE_HOLE,  s, e, f"Hole {s+1}{etag}"))
    for p in (0, 1, 2, 3):
        out.append(CaseLocation(TYPE_MASS, -1, p, f"Mass {MASS_PLATE_NAMES[p]}"))
    return out


def per_class_targets(total_per_type: int = 2000) -> List[Tuple[CaseLocation, int]]:
    """Return ``[(loc, n_samples), …]`` with totals balanced per type.

    Pristine uses the full ``total_per_type``; every other type splits its
    ``total_per_type`` across its locations as evenly as possible (residue
    spread to the first few locations of that type).
    """
    locs = enumerate_locations()
    by_type: dict[int, List[CaseLocation]] = {}
    for loc in locs:
        by_type.setdefault(loc.type_code, []).append(loc)

    targets: List[Tuple[CaseLocation, int]] = []
    for type_code, group in by_type.items():
        if type_code == TYPE_PRISTINE:
            targets.append((group[0], total_per_type))
            continue
        n_locs = len(group)
        base   = total_per_type // n_locs
        rem    = total_per_type - base * n_locs
        for i, loc in enumerate(group):
            n = base + (1 if i < rem else 0)
            targets.append((loc, n))
    return targets


def expand_samples(total_per_type: int = 2000) -> List[Tuple[int, CaseLocation]]:
    """Flatten to ``[(sample_id, loc), …]`` of length 5*total_per_type."""
    out: List[Tuple[int, CaseLocation]] = []
    sid = 0
    for loc, n in per_class_targets(total_per_type):
        for _ in range(n):
            out.append((sid, loc))
            sid += 1
    return out


if __name__ == "__main__":
    targets = per_class_targets()
    print(f"{'case':<24s} {'n':>5s}")
    print("-" * 32)
    by_type_total = {}
    for loc, n in targets:
        print(f"{loc.label:<24s} {n:>5d}")
        by_type_total[loc.type_code] = by_type_total.get(loc.type_code, 0) + n
    print("-" * 32)
    grand = 0
    for tc, n in by_type_total.items():
        print(f"  {TYPE_NAMES[tc]:<10s} total: {n}")
        grand += n
    print(f"  GRAND TOTAL: {grand}")
