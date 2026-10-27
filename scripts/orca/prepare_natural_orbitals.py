#!/usr/bin/env python3
"""
UNO / UCO restarts of the four field-free orbital-visualisation geometries.

For every <geometry>/<state> in angle_scans/visualization_orbitals/*/*/F0 this
writes angle_scans/natural_orbitals/<geometry>/<state>/ with
    <state>.inp     same level of theory, MORead from previous.gbw, ! UNO UCO
    previous.gbw    copy of the converged F0 wavefunction (job is self-contained)
    geometry.xyz    copy of the F0 geometry
so the tree can be submitted with submit_functional_scan.py (all jobs independent).
The source calculations are only read.
"""

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "visualisation"))
from orca_orbitals.orca_output import check_usable, parse_output  # noqa: E402

SOURCE = REPO / "runs/orca/short-2OS/angle_scans/visualization_orbitals"
TARGET = REPO / "runs/orca/short-2OS/angle_scans/natural_orbitals"
GEOMETRIES = ["M1", "L090_C080_R090", "L090_C120_R135", "L135_C120_R090"]
STATES = {"broken_symmetry": 1, "high_spin": 3}   # multiplicity

TEMPLATE = """! PBE0 def2-TZVP TightSCF UKS DEFGRID3 NoAutoStart MORead UNO UCO

%moinp "previous.gbw"

%scf
    MaxIter 500
    ConvCheckMode 0
end

%output
    Print[P_UNO_OccNum] 1
    Print[P_Hirshfeld] 1
end

%pal
    nprocs 16
end

%maxcore 2000

* xyzfile 0 {mult} geometry.xyz
"""


def main():
    for geom in GEOMETRIES:
        for state, mult in STATES.items():
            src = SOURCE / geom / state / "F0"
            out = parse_output(src / f"{state}.out")
            check_usable(out)                      # converged, terminated normally
            expected = "BS (Ms=0)" if state == "broken_symmetry" else "HS (Ms=1)"
            if out.electronic_state() != expected:
                raise RuntimeError(f"{src}: source is {out.electronic_state()}, expected {expected}")
            dst = TARGET / geom / state
            dst.mkdir(parents=True, exist_ok=True)
            (dst / f"{state}.inp").write_text(TEMPLATE.format(mult=mult))
            shutil.copy2(src / "geometry.xyz", dst / "geometry.xyz")
            shutil.copy2(src / f"{state}.gbw", dst / "previous.gbw")
            print(f"{geom}/{state}: {out.electronic_state()}, <S^2>={out.s2:.3f}, E={out.final_energy:.6f} Eh -> {dst}")


if __name__ == "__main__":
    main()
