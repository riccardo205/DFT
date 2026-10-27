#!/usr/bin/env python3
"""
Schematic of the molecule with the three constrained torsions (L, C, R) and
the four fragments (1-2-3-4) used in the angle-scan analysis.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # keep text as live text in the PDF
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList, get_connectivity_matrix, natural_cutoffs
from matplotlib.patches import Arc, FancyArrowPatch
from rdkit import Chem
from rdkit.Chem import rdDepictor


def build_adjacency(atoms):
    nl = NeighborList(natural_cutoffs(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)
    conn = get_connectivity_matrix(nl.nl, sparse=True).toarray()
    n = conn.shape[0]
    return {i: [j for j in range(n) if conn[i, j]] for i in range(n)}


def reachable_avoiding_edge(adj, start, edge):
    b, c = edge
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if (u, v) in ((b, c), (c, b)) or v in seen:
                continue
            seen.add(v)
            stack.append(v)
    return seen


def outer_fragment(adj, bond, core_markers):
    comp_b = reachable_avoiding_edge(adj, bond[0], bond)
    comp_c = reachable_avoiding_edge(adj, bond[1], bond)
    return comp_b if not (comp_b & set(core_markers)) else comp_c


def depict_2d(atoms, adj):
    """2D coordinates for the heavy-atom graph via RDKit; returns {atom_index: (x, y)}."""
    heavy = [i for i, a in enumerate(atoms) if a.symbol != "H"]
    new_idx = {old: k for k, old in enumerate(heavy)}
    rw = Chem.RWMol()
    for old in heavy:
        rw.AddAtom(Chem.Atom(atoms[old].symbol))
    for i in heavy:
        for j in adj[i]:
            if j > i and j in new_idx:
                rw.AddBond(new_idx[i], new_idx[j], Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    return {old: np.array([conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y]) for old, k in new_idx.items()}


def main():
    # ---- USER SETTINGS ----
    repo_root = Path(__file__).resolve().parent.parent
    geometry = repo_root / "scripts" / "orca" / "orca_functional_scan_setup" / "geometries" / "relaxed_main_conf_4.xyz"
    output = repo_root / "runs" / "orca" / "short-2OS" / "angle_scans" / "angle_scan_relaxation" / "plots" / "torsion_scheme.png"

    CENTER = (4, 9, 17, 18)
    LEFT = (42, 41, 0, 5)
    RIGHT = (21, 23, 28, 29)
    RADICAL_ATOMS = (41, 28)

    TORSIONS = {"L": LEFT, "C": CENTER, "R": RIGHT}
    THETA_LABELS = {"L": r"$\theta_L$", "C": r"$\theta_C$", "R": r"$\theta_R$"}
    TORSION_COLORS = {"L": "tab:blue", "C": "black", "R": "tab:red"}
    FRAGMENT_COLORS = {1: "#9ecae1", 2: "#c7e9c0", 3: "#fdd0a2", 4: "#f4a6a6"}

    atoms = read(geometry)
    adj = build_adjacency(atoms)
    n = len(atoms)

    core = set(CENTER)
    frag1 = outer_fragment(adj, (LEFT[1], LEFT[2]), core)
    frag4 = outer_fragment(adj, (RIGHT[1], RIGHT[2]), core)
    center_all = set(range(n)) - frag1 - frag4
    side_b = reachable_avoiding_edge(adj, CENTER[1], (CENTER[1], CENTER[2])) & center_all
    frag2, frag3 = side_b, center_all - side_b
    fragment_of = {}
    for k, frag in ((1, frag1), (2, frag2), (3, frag3), (4, frag4)):
        for i in frag:
            fragment_of[i] = k

    coords = depict_2d(atoms, adj)
    heavy = sorted(coords)
    xy = np.array([coords[i] for i in heavy])

    # Orient: principal axis horizontal, fragment 1 on the left, fragment 4 on the right.
    centered = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rot = vt.T
    for i in heavy:
        coords[i] = (coords[i] - xy.mean(axis=0)) @ rot
    c1 = np.mean([coords[i] for i in frag1 if i in coords], axis=0)
    c4 = np.mean([coords[i] for i in frag4 if i in coords], axis=0)
    if c1[0] > c4[0]:
        for i in heavy:
            coords[i] = coords[i] * np.array([-1.0, 1.0])
    if np.mean([coords[i][1] for i in frag2 | frag3 if i in coords]) < 0:
        pass  # vertical orientation is arbitrary; keep as is

    all_xy = np.array([coords[i] for i in heavy])
    bond_len = np.median([np.linalg.norm(coords[i] - coords[j]) for i in heavy for j in adj[i] if j > i and j in coords])

    x_lo, x_hi = all_xy[:, 0].min(), all_xy[:, 0].max()
    y_lo, y_hi = all_xy[:, 1].min(), all_xy[:, 1].max()
    x_mid = 0.5 * (x_lo + x_hi)
    LABEL_POSITIONS = {
        "L": (x_lo + 0.10 * (x_hi - x_lo), y_lo - 1.3 * bond_len),
        "C": (x_mid, y_hi + 1.5 * bond_len),
        "R": (x_lo + 0.90 * (x_hi - x_lo), y_hi + 1.5 * bond_len),
    }

    fig = plt.figure(figsize=(10, 7.2))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_aspect("equal")
    ax.axis("off")

    # Fragment halos
    for i in heavy:
        ax.scatter(*coords[i], s=900, color=FRAGMENT_COLORS[fragment_of[i]], zorder=1, linewidths=0)

    # Bonds
    for i in heavy:
        for j in adj[i]:
            if j > i and j in coords:
                ax.plot(*zip(coords[i], coords[j]), color="0.25", lw=1.6, zorder=2)

    # Atoms
    for i in heavy:
        ax.scatter(*coords[i], s=70, color="0.25", zorder=3)

    # Radical centers
    for r in RADICAL_ATOMS:
        ax.scatter(*coords[r], s=330, facecolors="none", edgecolors="gold", linewidths=3, zorder=4)

    # Torsions
    for name, quartet in TORSIONS.items():
        a, b, c, d = quartet
        color = TORSION_COLORS[name]
        pb, pc = coords[b], coords[c]
        ax.plot(*zip(pb, pc), color=color, lw=5, zorder=5, solid_capstyle="round")
        mid = (pb + pc) / 2
        axis_dir = (pc - pb) / np.linalg.norm(pc - pb)
        normal = np.array([-axis_dir[1], axis_dir[0]])
        # push the label away from the molecular centroid
        if np.dot(normal, mid - all_xy.mean(axis=0)) < 0:
            normal = -normal
        for atom in quartet:
            ax.scatter(*coords[atom], s=110, facecolors="white", edgecolors=color, linewidths=2.2, zorder=6)

        # curved double arrow across the bond = twist about B-C
        twist = FancyArrowPatch(
            mid + normal * 0.75 * bond_len, mid - normal * 0.75 * bond_len,
            connectionstyle="arc3,rad=0.9", arrowstyle="<|-|>", mutation_scale=16,
            color=color, lw=2.2, zorder=8,
        )
        ax.add_patch(twist)

        # label in the margin with a dashed leader line to the bond
        label_pos = np.array(LABEL_POSITIONS[name])
        ax.annotate(
            THETA_LABELS[name],
            xy=mid, xytext=label_pos, ha="center", va="center", fontsize=20, color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=1.4, ls="--", shrinkA=4, shrinkB=4),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.6), zorder=9,
        )

    pad = 1.6 * bond_len
    ax.set_xlim(all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ax.set_ylim(all_xy[:, 1].min() - 2.6 * bond_len, all_xy[:, 1].max() + 2.4 * bond_len)

    # Fragment numbers as large watermarks at each fragment's centroid
    for k, frag in ((1, frag1), (2, frag2), (3, frag3), (4, frag4)):
        pts = np.array([coords[i] for i in frag if i in coords])
        ax.text(pts[:, 0].mean(), pts[:, 1].mean(), str(k), ha="center", va="center", fontsize=54,
                fontweight="bold", color=FRAGMENT_COLORS[k], alpha=0.75, zorder=1.5)

    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=FRAGMENT_COLORS[k], label=f"fragment {k}") for k in (1, 2, 3, 4)]
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="gold",
                          markeredgewidth=3, markersize=13, label="radical center"))
    handles.append(Line2D([0], [0], color="0.2", lw=5, label="rotation axis"))
    ax.legend(handles=handles, loc="lower center", ncol=6, fontsize=10, frameon=False,
              bbox_to_anchor=(0.5, -0.02))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")
    print(f"Wrote {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
