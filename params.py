"""Local fallback ``params`` module for the LANL 3SBB reduced model.

The original ``reduced_model_semirigid.py`` imports ``params`` from
``pymodal/examples/los_alamos_3story``.  When that source tree is not
available (e.g. in a Colab session running directly from this repo) the
following constants reproduce the canonical 3SBB benchmark values so the
model still runs.  Values are taken from the IQS notebook header.
"""

# Plate (305 x 305 x 25.4 mm aluminium)
PLATE_LX = 0.305       # m
PLATE_LY = 0.305       # m
PLATE_LZ = 0.0254      # m

# Columns (25.4 x 6.4 mm aluminium)
COL_LX = 0.0254        # m  (wide dim, X-direction)
COL_LY = 0.0064        # m  (thin dim, Y-direction)

# Geometry
INTER_STOREY_GAP = 0.1524    # m   (storey height = PLATE_LZ + GAP = 0.1778)
COLUMN_GAP       = 0.0005    # m   plate-column gap
N_STORIES        = 3

# Aluminium properties
ALU_E   = 6.9e10            # Pa  (Young's modulus)
ALU_NU  = 0.33              # Poisson
ALU_RHO = 2700.0            # kg/m^3

# Damping & rail
MODAL_DAMPING  = 0.005
RAIL_DIRECTION = 'Y'
