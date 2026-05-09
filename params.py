"""Local params for the LANL 3SBB benchmark.

Mirrors the values in pymodal/examples/los_alamos_3story/params.py so that
the reduced-order model is self-contained.  Aluminum 6061-T6 is the LANL
specification; alternative alloys are listed in calibrate_3sbb_amplitude.py.
"""
# ── geometry ────────────────────────────────────────────────────────────────
PLATE_LX         = 0.305      # m
PLATE_LY         = 0.305      # m
PLATE_LZ         = 0.0254     # m  (1 inch plate thickness)
COL_LX           = 0.0254     # m  (column wide dimension)
COL_LY           = 0.0064     # m  (column thin dimension)
INTER_STOREY_GAP = 0.1524     # m  (free column length between plates)
COLUMN_GAP       = 0.0005     # m  (column-to-plate-edge clearance)
N_STORIES        = 3

# ── material (aluminum 6061-T6) ─────────────────────────────────────────────
ALU_E   = 6.89e10             # Pa     Young's modulus
ALU_NU  = 0.33                # —      Poisson ratio
ALU_RHO = 2700.0              # kg/m³  density

# ── modal damping (initial guess; per-mode damping is calibrated) ───────────
MODAL_DAMPING = 0.005

# ── boundary condition (base plate slides on a rail in Y) ───────────────────
RAIL_DIRECTION = 'Y'
