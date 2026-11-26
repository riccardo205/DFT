#!/usr/bin/env python3
"""
Two molecules (the same one twice, or two scan points, e.g. L045_C040_R045 above and L045_C040_R090 below), tilted
and stacked vertically (same drawing style as
molecule_torsion_view.py: carbons only, thin black bonds, atoms coloured by spin sublattice, spin arrows at the
radicals; no legend, no angle labels).

Orientation: as in molecule_torsion_view.py (mean anthracene plane in the screen, longest direction horizontal),
then tilted by EDGE_ON_ROT about the screen x axis (0 = flat, 90 = edge-on), so the molecule stays horizontal
and the two copies look stacked; far atoms are lighter and smaller (depth_fade, depth_scale in STYLE); then EXTRA_ROT (deg about screen x, y, z). The second copy is placed above the first,
the lower copy optionally upside down (FLIP_LOWER: 180 deg about the screen x axis),
separated by GAP (Å) between the highest point of the lower copy and the lowest point of the upper copy (arrows
included), and shifted horizontally by SHIFT_X (Å).

The lower molecule is put in the frame of the upper one by superimposing all carbons except the right outer
fragment (same left and central angles), so only the right fragment looks different.

Output: OUT_DIR/<upper>_<lower>_stack.pdf / .png at exactly SIZE_CM. Settings in the __main__ block.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import visualisation.paper.figure1.molecule_torsion_view as V  # noqa: E402


def extent(m, Y, arrow_len):
    """(min, max) of y over the carbons and the arrow tips."""
    y = np.concatenate([Y[m["carbons"], 1], [t[1] for t in V.arrow_tips(m, Y, arrow_len).values()]])
    return y.min(), y.max()


def kabsch_onto(X, Y_ref, idx):
    """X rigidly moved so that the atoms idx superimpose best on Y_ref[idx]."""
    P, Q = X[idx], Y_ref[idx]
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return (X - pc) @ R + qc


def main(geometry, out_file, geometry_lower, out_file_lower, out_dir, left, centre, right, radicals, edge_on_rot,
         extra_rot, flip_lower, gap, shift_x, size_cm, transparent, style, pad):
    m_up = V.prepare(geometry, out_file, left, centre, right, radicals)
    m_lo = V.prepare(geometry_lower, out_file_lower, left, centre, right, radicals, adj=m_up["adj"])
    own = V.adjacency(m_lo["atoms"])                  # report C-C contacts of the lower geometry not in the reference
    extra = sorted({tuple(sorted((i, j))) for i in own for j in own[i]} - {tuple(sorted((i, j))) for i in m_up["adj"]
                                                                          for j in m_up["adj"][i]})
    for i, j in extra:
        print(f"  [!] {geometry_lower.parent.name}: C{i}-C{j} = {m_lo['atoms'].get_distance(i, j):.2f} Å "
              f"(not a bond in {geometry.parent.name}; drawn with the reference bonds)")
    Y_up = V.oriented(m_up, (edge_on_rot, 0.0, 0.0))
    if any(extra_rot):                               # extra rotation about the screen axes of the tilted view
        c = Y_up[m_up["carbons"]].mean(axis=0)
        R = np.eye(3)
        for ax_name, deg in zip("xyz", extra_rot):
            R = V.rot(ax_name, deg) @ R
        Y_up = (Y_up - c) @ R.T + c
    # lower molecule in the same frame: superimpose everything except the right outer fragment (same L, C angles)
    outer_right = V.side_of_bond(m_up["adj"], right[2], (right[1], right[2]))
    common = [i for i in m_up["carbons"] if i not in outer_right]
    Y_lo = kabsch_onto(m_lo["X"], Y_up, common)
    rmsd = np.sqrt(((Y_lo[common] - Y_up[common]) ** 2).sum(axis=1).mean())
    if flip_lower:                                   # lower copy upside down: 180 deg about the screen x axis
        c = Y_lo[m_lo["carbons"]].mean(axis=0)
        Y_lo = (Y_lo - c) @ V.rot("x", 180).T + c
    lo1, hi1 = extent(m_lo, Y_lo, style["arrow_len"])
    lo2, hi2 = extent(m_up, Y_up, style["arrow_len"])
    Y_up = Y_up + np.array([shift_x, hi1 - lo2 + gap, 0.0])

    fig, ax = V.new_figure(size_cm)
    for m, Y in ((m_lo, Y_lo), (m_up, Y_up)):
        V.draw_molecule(ax, m, Y, style)
    pts = np.vstack([Y[m["carbons"], :2] for m, Y in ((m_lo, Y_lo), (m_up, Y_up))]
                    + [np.array(list(V.arrow_tips(m, Y, style["arrow_len"]).values()))
                       for m, Y in ((m_lo, Y_lo), (m_up, Y_up))])
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    name = geometry.parent.name if geometry_lower == geometry else f"{geometry.parent.name}_{geometry_lower.parent.name}"
    V.save(fig, out_dir / f"{name}_stack", transparent)
    print(f"upper {geometry.parent.name}, lower {geometry_lower.parent.name} (RMSD of the common part {rmsd:.2f} Å); "
          f"gap {gap} Å, shift {shift_x} Å, lower flipped: {flip_lower}")


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    REPO = HERE.parents[1]
    SCANS = REPO / "runs" / "orca" / "short-2OS" / "angle_scans" / "angle_scan_relaxation"
    UPPER = "L045_C040_R045"                    # upper molecule
    LOWERS = ["L045_C040_R090", "L045_C040_R135", "L045_C040_R000"]   # lower molecule, one figure each (same name as UPPER = two copies)
    OUT_DIR = HERE / "molecule_torsions"

    # 0-based atom indices (same as molecule_torsion_view.py)
    LEFT = (42, 41, 0, 5)
    CENTRE = (4, 9, 17, 18)
    RIGHT = (21, 23, 28, 29)
    RADICALS = (41, 28)

    EDGE_ON_ROT = 45.0                          # deg tilt about the screen x axis (0 = flat in the screen, 90 = edge-on;
                                                # negative tilts the other way)
    EXTRA_ROT = (0.0, 0.0, 0.0)                 # further rotation (deg) about screen x, y, z of the edge-on view
    FLIP_LOWER = False                          # lower copy upside down (180 deg about the screen x axis)
    GAP = 0.2                                   # Å between the two copies (arrows included)
    SHIFT_X = 2.0                               # Å, horizontal shift of the upper copy

    SIZE_CM = (4.0, 4.0)
    TRANSPARENT = True
    PAD = 0.4                                   # Å of empty margin
    STYLE = {"atom_size": 4.0, "atom_edge_lw": 0.15, "bond_lw": 0.4,
             "arrow_len": 1.6, "arrow_lw": 0.8, "arrow_head": 4,
             "col_a": "#C0392B", "col_b": "#1F4E9A",
             "depth_fade": 0.0,                 # 0..1: farthest atoms/bonds mixed with white (depth cue)
             "depth_scale": 0.3}                # 0..1: farthest atoms smaller

    for lower in LOWERS:                        # geometry.xyz + high_spin.out (Mulliken spins) of each scan point
        main(SCANS / UPPER / "geometry.xyz", SCANS / UPPER / "high_spin.out", SCANS / lower / "geometry.xyz",
             SCANS / lower / "high_spin.out", OUT_DIR, LEFT, CENTRE, RIGHT, RADICALS, EDGE_ON_ROT, EXTRA_ROT,
             FLIP_LOWER, GAP, SHIFT_X, SIZE_CM, TRANSPARENT, STYLE, PAD)
