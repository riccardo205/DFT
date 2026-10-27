#!/usr/bin/env python3
"""
Frontier / natural / corresponding orbital images from existing ORCA jobs.

Edit the CONFIGURATION block and run:   python render_orca_orbitals.py
Nothing is recomputed with ORCA: cubes come from the converged .gbw/.uno/.uco
via orca_plot, and the original calculation files are never modified.
See orca_orbitals/README.md for details.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orca_orbitals.pipeline import Settings, run  # noqa: E402
from orca_orbitals.render import RenderStyle  # noqa: E402

# ============================================================
# CONFIGURATION
# ============================================================
ROOT = Path(__file__).resolve().parents[1] / "runs/orca/short-2OS/dimers_run2/run2_broken_symmetry/relaxed/conf_7/"

RECURSIVE = True

ORBITALS = ["HOMO-1", "HOMO", "LUMO", "LUMO+1"]
SPINS = ["alpha", "beta"]

GRID_SIZE = 90              # points per axis passed to orca_plot
ISOVALUE = 0.02             # |psi| in bohr^-3/2, same for every image

RENDER_MO = True
RENDER_UNO = True           # used automatically when <job>.uno exists
RENDER_UCO = True           # used automatically when <job>.uco exists

OUTPUT_DIR_NAME = "orbital_analysis"
CENTRAL_OUTPUT = True       # True: all results in <ROOT>/<OUTPUT_DIR_NAME>/<job path>/
                            # False: results in <job dir>/<OUTPUT_DIR_NAME>/ next to each calculation

SMOKE_TEST = True           # True: process ONE calculation (the reference) to check settings first
SMOKE_TEST_CALC = None      # e.g. "M1/broken_symmetry/F0" to choose which one

# --- reproducible orientation ---------------------------------------------
ALIGN = True                # Kabsch-align every geometry onto the reference
# Rigid backbone used for alignment (0-based atom indices). Atoms 0-13 are the
# heavy atoms of fragment 2 in the short-2OS numbering (see analyze_field_sweep.py).
# None -> all heavy atoms (fine when only the field or spin state changes).
ALIGN_ATOMS = list(range(14))
REFERENCE = None            # path/label of the reference job; None -> first usable job
VIEW_ROTATION_DEG = (0.0, 0.0, 0.0)   # extra rotation of the display frame about x, y, z
FRAME_MARGIN = 3.0          # Angstrom of space around the atoms for the lobes
REUSE_SAVED_VIEW = True     # reuse <ROOT>/<OUTPUT_DIR_NAME>/view_settings.json if present

# --- natural / corresponding orbitals -------------------------------------
UNO_OCCUPATION_RANGE = (0.05, 1.95)   # candidate UNOs; closest to n = 1 are plotted first
UNO_MAX_ORBITALS = 6
UCO_MAX_OVERLAP = 0.95                # pairs with alpha/beta overlap below this are "magnetic"
UCO_MAX_PAIRS = 4

# --- misc -------------------------------------------------------------------
PHASE_CONVENTION = "none"   # "none" (sign as written by ORCA) or "largest_positive"
REGENERATE_CUBES = False
KEEP_CUBES = True           # 90^3 cube ~ 10 MB; set False to delete after rendering
ORCA_PLOT = None            # path to orca_plot; None -> PATH / $ORCA_DIR / ~/software/orca*
INCLUDE_PATTERNS = []       # fnmatch on .out path relative to ROOT, e.g. ["*/F0/*"]
EXCLUDE_PATTERNS = []
IGNORE_DIR_NAMES = []

STYLE = RenderStyle(
    window_size=(1600, 1200),
    background="white",
    positive_color="#1f4fd1",
    negative_color="#d62728",
    orbital_opacity=0.85,
    element_colors={"C": "#404040", "H": "#ffffff"},
    atom_radius_scale=0.40,
    bond_radius=0.09,
)


def main():
    settings = Settings(
        root=ROOT, recursive=RECURSIVE, orbitals=ORBITALS, spins=SPINS, grid_size=GRID_SIZE,
        isovalue=ISOVALUE, render_mo=RENDER_MO, render_uno=RENDER_UNO, render_uco=RENDER_UCO,
        output_dir_name=OUTPUT_DIR_NAME, central_output=CENTRAL_OUTPUT, orca_plot=ORCA_PLOT, regenerate_cubes=REGENERATE_CUBES,
        keep_cubes=KEEP_CUBES, smoke_test=SMOKE_TEST, smoke_test_calc=SMOKE_TEST_CALC,
        include_patterns=INCLUDE_PATTERNS, exclude_patterns=EXCLUDE_PATTERNS, ignore_dir_names=IGNORE_DIR_NAMES,
        align=ALIGN, align_atoms=ALIGN_ATOMS, reference=REFERENCE, view_rotation_deg=VIEW_ROTATION_DEG,
        frame_margin=FRAME_MARGIN, reuse_saved_view=REUSE_SAVED_VIEW, phase_convention=PHASE_CONVENTION,
        uno_occupation_range=UNO_OCCUPATION_RANGE, uno_max_orbitals=UNO_MAX_ORBITALS,
        uco_max_overlap=UCO_MAX_OVERLAP, uco_max_pairs=UCO_MAX_PAIRS, style=STYLE,
    )
    run(settings)


if __name__ == "__main__":
    main()
