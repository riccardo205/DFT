#!/usr/bin/env python3
"""
Stacked-dimer view of a real dimer geometry (e.g. run2 / conf_28), in the drawing style of
figure1/molecule_stack_view.py: carbons only, thin black bonds, atoms coloured by the sign of their Mulliken spin,
spin arrows at the four radicals; no legend, no labels.

Unlike molecule_stack_view.py (two separate monomer geometries stacked by hand), the two molecules keep their real
relative position: both come from one geometry file (monomer 1 = atoms 0 ... N_MONO-1, monomer 2 = the next
N_MONO, same numbering). Atom colours and arrows are the Mulliken spins of one broken-symmetry output (SPIN_OUT,
e.g. the lowest-energy determinant), so they show that spin configuration as it is (no sublattice flipping).

Orientation: z (towards the viewer) = stacking direction (centroid of molecule 1 -> centroid of molecule 2), x = radical-radical axis of molecule 1 (perpendicular to z); then tilted by EDGE_ON_ROT about the screen x axis
(as figure 1) so the two molecules look stacked, then EXTRA_ROT. EXPLODE (Å) optionally pushes the two
molecules apart along the stacking direction (0 = real geometry).

Output: OUT_DIR/<NAME>_stack.pdf / .png at exactly SIZE_CM. Settings in the __main__ block.
"""

import sys
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                          # DFT root: makes visualisation.* importable
import visualisation.paper.figure1.molecule_torsion_view as V  # noqa: E402


def monomer(atoms, adj_all, rho, off, n_mono, left, centre, right, radicals):
    """Drawing dict (as V.prepare) of the monomer made of atoms off ... off + n_mono - 1."""
    idx = set(range(off, off + n_mono))
    carbons = [i for i in idx if atoms[i].symbol == "C"]
    cset = set(carbons)
    adj = {i: (adj_all[i] & idx) if i in idx else set() for i in range(len(atoms))}
    L, C, R = [tuple(k + off for k in t) for t in (left, centre, right)]
    rads = [r + off for r in radicals]
    half_left = V.side_of_bond(adj, C[1], (C[1], C[2]))
    outer_left = V.side_of_bond(adj, L[1], (L[1], L[2]))
    outer_right = V.side_of_bond(adj, R[2], (R[1], R[2]))
    left_rad = next(r for r in rads if r in half_left)
    right_rad = next(r for r in rads if r not in half_left)
    sub = {i: (1 if rho[i] >= 0 else -1) for i in carbons}
    return {"adj": adj, "carbons": carbons, "sub": sub, "left_rad": left_rad, "right_rad": right_rad,
            "anth": ((half_left - outer_left) & cset) | (cset - half_left - outer_right)}


def main(geometry, spin_out, out_dir, name, n_mono, left, centre, right, radicals, edge_on_rot, extra_rot, explode,
         size_cm, transparent, style, pad, view_matrix=None, invert_spins=False):
    atoms = read(geometry)
    if len(atoms) != 2 * n_mono:
        print(f"  [!] {len(atoms)} atoms, expected 2 x {n_mono}")
    adj_all = V.adjacency(atoms)
    rho = V.mulliken_spins(spin_out, len(atoms))
    if rho is None:
        raise SystemExit(f"no Mulliken spins in {spin_out}")
    if invert_spins:                         # global spin flip (dudu -> udud): same state, all arrows reversed
        rho = -rho
    ms = [monomer(atoms, adj_all, rho, k * n_mono, n_mono, left, centre, right, radicals) for k in (0, 1)]
    X = atoms.get_positions()

    # stacking direction = centroid of molecule 1 -> centroid of molecule 2 (the twisted molecules have no single
    # plane); push the molecules apart along it (EXPLODE)
    allc = ms[0]["carbons"] + ms[1]["carbons"]
    c1, c2 = (X[m["carbons"]].mean(axis=0) for m in ms)
    stack_d = np.linalg.norm(c2 - c1)
    z = (c2 - c1) / stack_d
    X = X.copy()
    X[n_mono:] += explode * z
    # frame: z = stacking direction, x = longest direction of the dimer perpendicular to it (left radical of
    # molecule 1 on the left), then the tilt about the screen x axis and EXTRA_ROT
    x = X[ms[0]["right_rad"]] - X[ms[0]["left_rad"]]           # radical-radical axis of molecule 1
    x -= (x @ z) * z
    x /= np.linalg.norm(x)
    if (X[ms[0]["right_rad"]] - X[ms[0]["left_rad"]]) @ x < 0:
        x = -x
    R = np.array([x, np.cross(z, x), z])
    for ax_name, deg in zip("xyz", (edge_on_rot, 0, 0)):
        R = V.rot(ax_name, deg) @ R
    if view_matrix is not None:              # fixed view: rows = screen right, up, towards the viewer (file axes)
        R = np.asarray(view_matrix, float)
    for ax_name, deg in zip("xyz", extra_rot):
        R = V.rot(ax_name, deg) @ R
    Y = (X - X[allc].mean(axis=0)) @ R.T

    fig, ax = V.new_figure(size_cm)
    for m in ms:
        V.draw_molecule(ax, m, Y, style)
    pts = np.vstack([Y[allc, :2]] + [np.array(list(V.arrow_tips(m, Y, style["arrow_len"]).values())) for m in ms])
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    out_dir.mkdir(parents=True, exist_ok=True)
    V.save(fig, out_dir / f"{name}_stack", transparent)
    spins = {r: rho[r] for m in ms for r in (m["left_rad"], m["right_rad"])}
    upper = 1 if Y[ms[1]["carbons"], 1].mean() > Y[ms[0]["carbons"], 1].mean() else 0
    print(f"{name}: spins from {Path(spin_out).name}; radical spins " +
          ", ".join(f"C{r}: {v:+.2f}" for r, v in spins.items()) +
          f"; stacking distance {stack_d:.2f} Å (+{explode} Å explode); upper on screen: monomer {upper + 1}")


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    REPO = HERE.parents[2]
    RUN = REPO / "runs" / "orca" / "short-2OS" / "dimers_run2"
    CONF = "conf_28"
    GEOMETRY = RUN / "run2" / CONF / "high_spin" / "high_spin.xyz"          # relaxed high-spin dimer geometry
    SPIN_OUT = RUN / "selected_out_files" / f"run2_{CONF}" / "dudu.out"    # BS ground state of conf_28 (lowest E)
    NAME = f"run2_{CONF}"
    INVERT_SPINS = True                         # flip every spin (e.g. dudu drawn as udud)
    OUT_DIR = HERE / "outputs"

    # monomer atoms and 0-based indices (same as figure1; monomer 2 = the same + N_MONO)
    N_MONO = 86
    LEFT = (42, 41, 0, 5)
    CENTRE = (4, 9, 17, 18)
    RIGHT = (21, 23, 28, 29)
    RADICALS = (41, 28)

    # view: VIEW_MATRIX rows = screen right, screen up, towards the viewer, in the axes of the xyz file. This one
    # reproduces the molecular-viewer screenshot figure4/outputs/'Screenshot 2026-11-24 at 17.00.16.png'
    # (fitted on its outline). None: automatic view (z = stacking direction, tilted by EDGE_ON_ROT as figure 1).
    VIEW_MATRIX = [[-0.3724, -0.2355, 0.8977],
                   [-0.4559, -0.7961, -0.3980],
                   [0.8084, -0.5575, 0.1891]]
    EDGE_ON_ROT = 45.0                          # deg tilt about the screen x axis (automatic view only)
    EXTRA_ROT = (0.0, 0.0, 50.0)                # further rotation (deg) about screen x, y, z (both views);
                                                # z = 90: screenshot view turned 90 deg anticlockwise (top -> left)
    EXPLODE = 0.0                               # Å, extra separation along the stacking direction (0 = real geometry;
                                                # > 0 = "exploded" view, not to scale between the molecules)

    SIZE_CM = (3.6, 3.6)
    TRANSPARENT = True
    PAD = 0.4                                   # Å of empty margin
    STYLE = {"atom_size": 4.0, "atom_edge_lw": 0.15, "bond_lw": 0.4,
             "arrow_len": 1.6, "arrow_lw": 0.8, "arrow_head": 4,
             "col_a": "#C0392B", "col_b": "#1F4E9A",   # spin up / down
             "depth_fade": 0.0, "depth_scale": 0.3,
             "arrow_depth": False}                # False: spin arrows on top (two radicals sit inside the
                                                  # edge-on central units in this view and would be hidden)

    main(GEOMETRY, SPIN_OUT, OUT_DIR, NAME, N_MONO, LEFT, CENTRE, RIGHT, RADICALS, EDGE_ON_ROT, EXTRA_ROT, EXPLODE,
         SIZE_CM, TRANSPARENT, STYLE, PAD, VIEW_MATRIX, INVERT_SPINS)
