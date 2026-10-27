#!/usr/bin/env python3
"""
Orbitals involved in the SF-TDA calculations (sf_tda_validation/<geometry>/<functional>/sf_tda.gbw).

SF-TDA starts from the Ms = 1 UKS triplet. The low-lying spin-flip roots are combinations of
    alpha SOMO (177a = HOMO-1, 178a = HOMO)  ->  beta 177b (= LUMO), 178b (= LUMO+1)
so these four orbitals per calculation are what the root compositions in
sf_tda_validation_results/sf_tda_roots.csv refer to. Uses the existing orca_orbitals pipeline
(orca_plot cubes from the converged .gbw, nothing is recomputed); the display frame is the one
saved for visualization_orbitals, so orientations match the earlier BS/HS images.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orca_orbitals.pipeline import Settings, run  # noqa: E402
from render_orca_orbitals import STYLE  # noqa: E402

SCANS = Path(__file__).resolve().parents[1] / "runs/orca/short-2OS/angle_scans"
ROOT = SCANS / "sf_tda_validation"
OUTPUT_DIR_NAME = "orbital_analysis"
SAVED_VIEW = SCANS / "visualization_orbitals" / "orbital_analysis" / "view_settings.json"


def main():
    out = ROOT / OUTPUT_DIR_NAME
    out.mkdir(exist_ok=True)
    if SAVED_VIEW.exists() and not (out / "view_settings.json").exists():
        shutil.copy2(SAVED_VIEW, out / "view_settings.json")
    settings = Settings(
        root=ROOT, recursive=True,
        orbitals=["HOMO-1", "HOMO", "LUMO", "LUMO+1"], spins=["alpha", "beta"],
        grid_size=90, isovalue=0.02,
        render_mo=True, render_uno=False, render_uco=False,
        output_dir_name=OUTPUT_DIR_NAME, central_output=True, orca_plot=None,
        regenerate_cubes=False, keep_cubes=True, smoke_test=False, smoke_test_calc=None,
        include_patterns=["*/sf_tda.out"], exclude_patterns=[], ignore_dir_names=[],
        align=True, align_atoms=list(range(14)), reference=None, view_rotation_deg=(0.0, 0.0, 0.0),
        frame_margin=3.0, reuse_saved_view=True, phase_convention="none",
        uno_occupation_range=(0.05, 1.95), uno_max_orbitals=6, uco_max_overlap=0.95, uco_max_pairs=4,
        style=STYLE,
    )
    run(settings)


if __name__ == "__main__":
    main()
