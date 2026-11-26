#!/usr/bin/env python3
"""
Same view and window as fragment_overlap_view.py (top view of the two fluorenyls of the dominant intermolecular J,
red / blue spin dots, black bonds, spin arrows), with the two magnetic orbitals of that pair drawn underneath as
isosurfaces -- as visualisation/dimer_overlap/magnetic_orbitals.py, but:
  * only the main fragment: each orbital is kept only on the grid points whose nearest heavy atom belongs to one
    of the two main fluorenyls (MAIN_ONLY), so the lobes on the rest of the dimer disappear;
  * lighter tones (ORB_COL, pastel) and partly transparent (OPACITY);
  * FACING_ONLY: only the lobes between the two planes (orbital 1 above the first fluorenyl, orbital 2
    below the second), i.e. the lobes that overlap; seen from the top the two lobes of a pi orbital
    would otherwise lie on each other and blend.
Orbitals: the magnetic UCO pair of the BS state with the second radical flipped (magnetic_orbitals.pick_state),
from the cubes cached in visualisation/output/orca/dimers_run2/dimer_overlap/<run>_<dimer>/magnetic_orbitals/cubes
(written by magnetic_orbitals.py; missing ones are made with orca_plot). The sign of orbital 2 is chosen so that
int psi1 psi2 >= 0 (ALIGN_PHASE).
The isosurfaces are rendered with pyvista (orthographic, looking down z, exactly the drawing window) to a
transparent image placed under the matplotlib drawing of the structure.

Output: OUT_DIR/<run>_<dimer>_orbitals.pdf / .png (same size as the overlap panels). Settings in __main__.
"""

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "visualisation"))
sys.path.insert(0, str(REPO / "visualisation" / "dimer_overlap"))
import visualisation.paper.figure1.molecule_torsion_view as V  # noqa: E402
from visualisation.paper.figure4.fragment_overlap_view import draw_structure, new_axes, scene  # noqa: E402
from magnetic_orbitals import (facing, frame, half_weights, lobe_mesh, magnetic_candidates,  # noqa: E402
                               make_cubes, pick_state)
from orca_orbitals.cube import BOHR_TO_ANGSTROM, read_cube  # noqa: E402
from orca_orbitals.isosurface import extract_lobes  # noqa: E402


def magnetic_pair(key, dimers_root, cube_root, a, b, grid, spin_config):
    """(cube of orbital 1 on a, cube of orbital 2 on b, state, UCO index, S_UCO)."""
    run, dimer = key.split("/", 1)
    d = dimers_root / f"{run}_broken_symmetry" / "relaxed" / dimer
    cfg, on_alpha, on_beta, o = pick_state(d, a, b, spin_config)
    halves = {k: set(v) for k, v in json.loads((d / "radical_fragments.json").read_text())["fragments"].items()}
    best = None
    for i, s_uco in magnetic_candidates(o, cfg):
        paths = make_cubes(d / cfg / f"{cfg}.uco", i, grid, cube_root / key.replace("/", "_") / "magnetic_orbitals"
                           / "cubes")
        ca, cb = read_cube(paths[0]), read_cube(paths[1])
        score = half_weights(ca, halves)[on_alpha] * half_weights(cb, halves)[on_beta]
        if best is None or score > best[0]:
            best = (score, i, s_uco, ca, cb)
    _, i, s_uco, ca, cb = best
    cubes = {on_alpha: ca, on_beta: cb}
    return cubes[a], cubes[b], cfg, i, s_uco


def main_only(cube, keep_atoms):
    """Copy of the cube with psi = 0 on the grid points whose nearest heavy atom is not in keep_atoms."""
    heavy = np.where(cube.atomic_numbers > 1)[0]
    owner = heavy[cKDTree(cube.positions[heavy]).query(cube.cartesian_grid().reshape(-1, 3))[1]]
    mask = np.isin(owner, sorted(keep_atoms)).reshape(cube.shape)
    return replace(cube, data=np.where(mask, cube.volume(), 0.0))


def render_lobes(lobes, window, z_mid, width_px, opacity, png):
    """Orthographic render looking down z of exactly the window (x0, x1, y0, y1), transparent background."""
    x0, x1, y0, y1 = window
    win = [int(width_px), int(round(width_px * (y1 - y0) / (x1 - x0)))]
    pl = pv.Plotter(off_screen=True, window_size=win, lighting="light_kit")
    for m, c in lobes:
        pl.add_mesh(m, color=c, opacity=opacity, smooth_shading=True, specular=0.15, ambient=0.45, diffuse=0.6)
    pl.enable_depth_peeling(number_of_peels=12)
    pl.enable_anti_aliasing("ssaa")
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2, z_mid])
    pl.camera_position = [tuple(centre + [0, 0, 100]), tuple(centre), (0, 1, 0)]
    pl.enable_parallel_projection()
    pl.camera.parallel_scale = (y1 - y0) / 2
    pl.camera.clipping_range = (1.0, 300.0)
    pl.screenshot(str(png), transparent_background=True)
    pl.close()


def main(key, dimers_root, csv_root, cube_root, out_dir, spin_config, invert_spins, y_max, pad, width_cm,
         transparent, st, context_alpha, arrows_context, window_a, grid, orb_spin_config, isovalue, opacity,
         orb_col, main_only_flag, align_phase, width_px, facing_only=True, arrow_outline=None):
    sc = scene(key, dimers_root, csv_root, spin_config, invert_spins, y_max, pad, st, window_a)
    a, b, fl = sc["a"], sc["b"], sc["fl"]
    c1, c2, cfg, i, s_uco = magnetic_pair(key, dimers_root, cube_root, a, b, grid, orb_spin_config)
    dv = abs(np.linalg.det(c1.axes))
    s_grid = float((c1.volume() * c2.volume()).sum() * dv)
    phase2 = -1.0 if align_phase and s_grid < 0 else 1.0
    if main_only_flag:
        c1, c2 = main_only(c1, sc["main_atoms"]), main_only(c2, sc["main_atoms"])

    # same frame as the structure (inspect_dimer: fluorenyl of a flat in xy, b above); check it matches
    pos = c1.positions * BOHR_TO_ANGSTROM
    cP, R = frame(pos, fl[a][0], fl[a][1], fl[b][0])
    dev = np.abs((pos - cP) @ R.T - sc["X"]).max()
    if dev > 1e-3:
        print(f"  [!] orbital frame differs from the structure frame by {dev:.3f} Å")

    lobes = []
    for k, (cube, ph) in enumerate(((c1, 1.0), (c2, phase2)), 1):
        lb, notes, _ = extract_lobes(cube, isovalue, phase=ph)
        for note in notes:
            print(f"  [orbital {k}] {note}")
        for name, col in zip(("positive", "negative"), orb_col[k]):
            if lb[name] is None:
                continue
            m = lobe_mesh((lb[name].vertices - cP) @ R.T, lb[name].faces)
            if facing_only:                  # orbital 1 above the plane of a, orbital 2 below the plane of b
                m = facing(m, 0.0, True) if k == 1 else facing(m, sc["desc"]["d_perp_A"], False)
            if m.n_points:
                lobes.append((m, col))

    fig, ax, h_cm = new_axes(sc["window"], width_cm)
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "lobes.png"
        render_lobes(lobes, sc["window"], 0.5 * sc["desc"]["d_perp_A"], width_px, opacity, png)
        x0, x1, y0, y1 = sc["window"]
        ax.imshow(plt.imread(png), extent=(x0, x1, y0, y1), origin="upper", zorder=1, interpolation="bilinear")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    draw_structure(ax, sc, st, context_alpha, arrows_context, z_base=2.0, arrow_outline=arrow_outline)
    out_dir.mkdir(parents=True, exist_ok=True)
    V.save(fig, out_dir / f"{key.replace('/', '_')}_orbitals", transparent)
    print(f"{key}: {sc['desc']['pair']} ({sc['desc']['J_meV']:+.2f} meV); orbitals of {cfg}, UCO {i}, "
          f"S_UCO = {s_uco:.4f}, int psi1 psi2 = {s_grid * phase2:+.4f} (full orbitals); iso ±{isovalue}; "
          f"{width_cm:.2f} x {h_cm:.2f} cm")


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    KEYS = ["run2/conf_28", "run2/conf_7", "run2/conf_11"]
    DIMERS_ROOT = REPO / "runs" / "orca" / "short-2OS" / "dimers_run2"
    CSV_ROOT = REPO / "visualisation" / "output" / "orca" / "dimers_run2"
    CUBE_ROOT = CSV_ROOT / "dimer_overlap"      # <run>_<dimer>/magnetic_orbitals/cubes (from magnetic_orbitals.py)
    OUT_DIR = HERE / "outputs"

    # structure and window: as fragment_overlap_view.py (keep these equal to get the same panels)
    SPIN_CONFIG = None
    INVERT_SPINS = True
    Y_MAX = {"run2/conf_11": None, "default": 4.0}
    WINDOW_A = (9.0, 8.2)
    PAD = 0.6
    WIDTH_CM = 3.1
    TRANSPARENT = True
    CONTEXT_ALPHA = 0.25
    ARROWS_CONTEXT = False
    STYLE = {"atom_size": 9.0, "atom_edge_lw": 0.2, "bond_lw": 0.6,
             "arrow_len": 1.0, "arrow_lw": 0.9, "arrow_head": 4.5,
             "col_a": "#C0392B", "col_b": "#1F4E9A"}

    # orbitals
    GRID = 100                                  # orca_plot grid of the cached cubes
    ORB_SPIN_CONFIG = None                      # BS state of the magnetic orbitals; None = second radical flipped
    ISOVALUE = 0.02                             # +/- amplitude of the isosurfaces
    OPACITY = 1                              # isosurface opacity (lower = lighter)
    ORB_COL = {1: ("#8DB0EE", "#BFE8FA"),       # orbital 1 (first fluorenyl): + light blue / - pale cyan
               2: ("#EE9A9C", "#FBE6A2")}       # orbital 2 (second fluorenyl): + light red / - pale yellow
    MAIN_ONLY = True                            # keep the orbitals only on the two main fluorenyls
    FACING_ONLY = False                          # only the lobes between the two planes (the ones that overlap);
                                                # False: both lobes of each pi orbital (they blend seen from top)
    ALIGN_PHASE = True                          # sign of orbital 2 such that int psi1 psi2 >= 0
    ARROW_OUTLINE = 0.3                         # pt, white contour around the spin arrows (None: none)
    WIDTH_PX = 1600                             # resolution of the isosurface image

    for k in KEYS:
        main(k, DIMERS_ROOT, CSV_ROOT, CUBE_ROOT, OUT_DIR, SPIN_CONFIG, INVERT_SPINS, Y_MAX, PAD, WIDTH_CM,
             TRANSPARENT, STYLE, CONTEXT_ALPHA, ARROWS_CONTEXT, WINDOW_A, GRID, ORB_SPIN_CONFIG, ISOVALUE, OPACITY,
             ORB_COL, MAIN_ONLY, ALIGN_PHASE, WIDTH_PX, FACING_ONLY, ARROW_OUTLINE)
