#!/usr/bin/env python3
"""
Electric-field sweep on fixed geometries, HS and BS, with MO continuation.

Structures (geometry frozen, same geometry for HS and BS):
    L090_C080_R090 : symmetric, both outer 90 deg, dE ~ +0.1 meV (no built-in asymmetry)
    L135_C120_R090 : asymmetric, most negative dE (-0.31 meV)
    L090_C120_R135 : approximate mirror of the above (-0.17 meV)
    M1             : free-relaxation global-minimum candidate (AFM, well conjugated)

Directions (per-structure unit vectors, in that geometry's own Cartesian frame):
    axis : e_14   = unit(centroid(frag4) - centroid(frag1))
    perp : e_perp = component of (centroid(frag2+frag3) - midpoint(frag1, frag4))
                    orthogonal to e_14, normalised. Points from the 1-4 line
                    towards the backbone (the molecule is U-shaped; this is also
                    its dominant out-of-axis principal direction).
    e_n = e_14 x e_perp is written to field_axes.csv but not run by default.

Field convention: the ORCA input gets "EField F*e" with F in a.u. So F > 0 on
"axis" means the EField vector points along fragment 1 -> fragment 4. Which way
the electrons move for a given sign is read off from q1 - q4 at small |F|; it is
not assumed here.

Continuation: per structure and state there is a zero-field point F0. Points
with |F| <= PARALLEL_MAX read the F0 .gbw and run in parallel; the larger
fields of each (direction, sign) are chained 0.001 -> 0.002 -> 0.005, each point
reading the previous point's .gbw (MORead). F0 reads
the converged PBE0 orbitals from angle_scan_exchange_analysis when available
(same method, same geometry), so it stays on the same HS/BS branch that
exchange_analysis.md analysed. M1 has no PBE0 run yet: F0 starts fresh
(GuessMix for BS).
"""

import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from ase.io import read

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "visualisation"))
from analyze_angle_scan import (  # noqa: E402
    build_adjacency,
    outer_terminal_fragment,
    split_fragment_at_bond,
)


def point_name(direction, field):
    if field == 0.0:
        return "F0"
    return f"{direction}_{'p' if field > 0 else 'm'}{abs(field):g}"


def vec_str(v):
    return ", ".join(f"{x:.8f}" for x in v)


def render_input(state, field_vec, moread, nprocs, maxcore_mb, max_iter):
    keywords = ["PBE0", "def2-TZVP", "TightSCF", "UKS", "DEFGRID3", "NoAutoStart"]
    if moread:
        keywords.append("MORead")
    elif state["guess_mix"]:
        keywords.append("GuessMix")

    text = "! " + " ".join(keywords) + "\n\n"
    if moread:
        text += '%moinp "guess.gbw"\n\n'
    text += (
        "%scf\n"
        f"    MaxIter {max_iter}\n"
        "    ConvCheckMode 0\n"
    )
    if field_vec is not None:
        text += f"    EField {vec_str(field_vec)}\n"
    text += (
        "end\n\n"
        "%output\n"
        "    Print[P_Hirshfeld] 1\n"
        "end\n\n"
        "%pal\n"
        f"    nprocs {nprocs}\n"
        "end\n\n"
        f"%maxcore {maxcore_mb}\n\n"
        f"* xyzfile 0 {state['multiplicity']} geometry.xyz\n"
    )
    return text


def main():
    # USER SETTINGS
    SCANS = REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans"
    EXCHANGE_ROOT = SCANS / "angle_scan_exchange_analysis"
    FREE_ROOT = SCANS / "angle_scan_free_relaxation"
    OUTPUT_ROOT = SCANS / "angle_scan_field_sweep"
    BASELINE_XYZ = REPO_ROOT / "scripts" / "orca" / "orca_functional_scan_setup" / "geometries" / "relaxed_main_conf_4.xyz"

    # name -> (geometry, {state_folder: seed gbw or None})
    STRUCTURES = {
        "L090_C080_R090": EXCHANGE_ROOT / "L090_C080_R090",
        "L135_C120_R090": EXCHANGE_ROOT / "L135_C120_R090",
        "L090_C120_R135": EXCHANGE_ROOT / "L090_C120_R135",
        # M1: lowest converged run assigned to M1 (BS-relaxed, E_vs_global = 0.0 meV)
        "M1": FREE_ROOT / "L045_C080_R045" / "broken_symmetry" / "broken_symmetry.xyz",
    }

    DIRECTIONS = ["axis", "perp"]       # add "normal" to also sweep along e_14 x e_perp
    # Magnitudes in a.u., in the order the branch walks them (continuation).
    # 0.005 is the wide stress test; it is last so it cannot spoil the others.
    FIELD_MAGNITUDES = [0.00025, 0.0005, 0.001, 0.002, 0.005]
    # Up to this |F| every point starts directly from the F0 orbitals and runs in
    # parallel (dV across 1-4 <~ 0.5 eV: smooth polarisation). Above it, points are
    # chained from the previous magnitude to stay on the same HS/BS branch.
    PARALLEL_MAX = 0.001

    NPROCS = 16
    MAXCORE_MB = 2000
    MAX_ITER = 500
    COPY_SEED_GBW = True
    OVERWRITE = False

    CENTER, LEFT, RIGHT = (4, 9, 17, 18), (42, 41, 0, 5), (21, 23, 28, 29)

    STATES = [
        {"folder": "high_spin", "multiplicity": 3, "guess_mix": False},
        {"folder": "broken_symmetry", "multiplicity": 1, "guess_mix": True},
    ]

    # ---- fragments from the clean baseline (same definition as analyze_exchange.py) ----
    ref = read(BASELINE_XYZ)
    adj = build_adjacency(ref)
    frag1 = outer_terminal_fragment(adj, (LEFT[1], LEFT[2]), set(CENTER))
    frag4 = outer_terminal_fragment(adj, (RIGHT[1], RIGHT[2]), set(CENTER))
    center_all = sorted(set(range(len(ref))) - set(frag1) - set(frag4))
    frag2, frag3 = split_fragment_at_bond(adj, (CENTER[1], CENTER[2]), center_all)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    axes_rows, manifest_rows = [], []

    for name, source in STRUCTURES.items():
        geom_file = source / "high_spin" / "geometry.xyz" if source.is_dir() else source
        atoms = read(geom_file)
        if atoms.get_chemical_symbols() != ref.get_chemical_symbols():
            raise ValueError(f"{name}: atom ordering differs from baseline")

        pos = atoms.positions
        c1, c4 = pos[frag1].mean(axis=0), pos[frag4].mean(axis=0)
        c23 = pos[frag2 + frag3].mean(axis=0)
        d14 = np.linalg.norm(c4 - c1)
        e14 = (c4 - c1) / d14
        bow = c23 - 0.5 * (c1 + c4)
        bow -= bow.dot(e14) * e14
        eperp = bow / np.linalg.norm(bow)
        enorm = np.cross(e14, eperp)
        unit = {"axis": e14, "perp": eperp, "normal": enorm}

        axes_rows.append({
            "structure": name,
            "geometry_source": str(geom_file.relative_to(REPO_ROOT)),
            "d_14_A": f"{d14:.4f}",
            "backbone_offset_A": f"{np.linalg.norm(bow):.4f}",
            **{f"{k}_{c}": f"{v[i]:.8f}" for k, v in unit.items() for i, c in enumerate("xyz")},
        })

        struct_dir = OUTPUT_ROOT / name
        struct_dir.mkdir(exist_ok=True)
        shutil.copy2(geom_file, struct_dir / "geometry.xyz")

        for state in STATES:
            state_dir = struct_dir / state["folder"]
            create = OVERWRITE or not state_dir.exists()
            if not create:
                print(f"Existing: {name}/{state['folder']} (inputs kept, chain files rewritten)")
            elif state_dir.exists():
                shutil.rmtree(state_dir)
            state_dir.mkdir(exist_ok=True)

            seed = source / state["folder"] / f"{state['folder']}.gbw" if source.is_dir() else None
            seed_ok = COPY_SEED_GBW and seed is not None and seed.exists()

            # (point, previous point, field vector) in execution order, one list per job.
            # |F| <= PARALLEL_MAX: one job per point, guess from F0 (all in parallel).
            # |F| >  PARALLEL_MAX: one continuation job per branch, starting from the
            #                     largest parallel point.
            chains = {"F0": [("F0", None, None)]}
            for direction in DIRECTIONS:
                for sign, tag in ((1, "p"), (-1, "m")):
                    prev, tail = "F0", []
                    for mag in FIELD_MAGNITUDES:
                        pt = point_name(direction, sign * mag)
                        fvec = sign * mag * unit[direction]
                        if mag <= PARALLEL_MAX:
                            chains[pt] = [(pt, "F0", fvec)]
                        else:
                            tail.append((pt, prev, fvec))
                        prev = pt
                    if tail:
                        chains[f"{direction}_{tag}_tail"] = tail

            for old in state_dir.glob("chain_*.txt"):
                old.unlink()
            for branch, chain in chains.items():
                with (state_dir / f"chain_{branch}.txt").open("w") as fh:
                    for pt, prev, fvec in chain:
                        fh.write(f"{pt} {prev or '-'}\n")
                        pdir = state_dir / pt
                        inp = pdir / f"{state['folder']}.inp"
                        if create:
                            pdir.mkdir()
                            shutil.copy2(geom_file, pdir / "geometry.xyz")
                            moread = prev is not None or seed_ok
                            if pt == "F0" and seed_ok:
                                shutil.copy2(seed, pdir / "guess.gbw")
                            inp.write_text(render_input(state, fvec, moread, NPROCS, MAXCORE_MB, MAX_ITER))

                        direction = "none" if pt == "F0" else pt.split("_")[0]
                        f_signed = 0.0 if fvec is None else float(np.dot(fvec, unit[direction]))
                        manifest_rows.append({
                            "structure": name,
                            "state": state["folder"],
                            "multiplicity": state["multiplicity"],
                            "job": branch,
                            "point": pt,
                            "previous": prev or ("seed:" + str(seed.relative_to(REPO_ROOT)) if seed_ok else "fresh"),
                            "direction": direction,
                            "field_au": f"{f_signed:g}",
                            "Fx": f"{0.0 if fvec is None else fvec[0]:.8f}",
                            "Fy": f"{0.0 if fvec is None else fvec[1]:.8f}",
                            "Fz": f"{0.0 if fvec is None else fvec[2]:.8f}",
                            "input_file": str(inp.relative_to(OUTPUT_ROOT)),
                        })
            if create:
                print(f"Created: {name}/{state['folder']}  (F0 seed: {'exchange_analysis gbw' if seed_ok else 'fresh guess'})")

    with (OUTPUT_ROOT / "field_axes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=axes_rows[0].keys())
        w.writeheader()
        w.writerows(axes_rows)

    if manifest_rows:
        with (OUTPUT_ROOT / "manifest.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=manifest_rows[0].keys())
            w.writeheader()
            w.writerows(manifest_rows)

    for f in ("run_chain.sh", "submit_field_sweep.sh", "clean_failed_points.sh"):
        shutil.copy2(SCRIPT_DIR / f, OUTPUT_ROOT / f)

    print()
    print("=" * 70)
    print("Electric-field sweep prepared")
    print("=" * 70)
    print(f"Runs      : {OUTPUT_ROOT}")
    print(f"Points    : {len(manifest_rows)}")
    print(f"Axes      : {OUTPUT_ROOT / 'field_axes.csv'}")


if __name__ == "__main__":
    main()
