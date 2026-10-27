#!/usr/bin/env python3
"""
Visual check of the field directions used in angle_scan_field_sweep.

The field vectors are read back from the generated ORCA inputs
(<struct>/broken_symmetry/{axis,perp}_p0.001/broken_symmetry.inp), not
recomputed. The axis arrow starts at the fragment-1 centroid with length
d(1-4): if the input is right, its tip lands on the fragment-4 centroid in every
view. Views are the molecule's own principal axes (independent of the field
vectors): top = PC1-PC2, side = PC1-PC3.
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_angle_scan import (  # noqa: E402
    build_adjacency,
    outer_terminal_fragment,
    split_fragment_at_bond,
)

EFIELD_RE = re.compile(r"EField\s+([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+)")
FRAG_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
INK, MUTED = "#0b0b0b", "#52514e"


def read_efield(inp):
    m = EFIELD_RE.search(inp.read_text())
    return np.array([float(x) for x in m.groups()])


def main():
    repo_root = Path(__file__).resolve().parent.parent
    sweep = repo_root / "runs" / "orca" / "short-2OS" / "angle_scans" / "angle_scan_field_sweep"
    baseline_xyz = repo_root / "scripts" / "orca" / "orca_functional_scan_setup" / "geometries" / "relaxed_main_conf_4.xyz"
    out_dir = sweep / "plots"
    out_dir.mkdir(exist_ok=True)

    CENTER, LEFT, RIGHT = (4, 9, 17, 18), (42, 41, 0, 5), (21, 23, 28, 29)
    STRUCTURES = ["L090_C080_R090", "L135_C120_R090", "L090_C120_R135", "M1"]

    ref = read(baseline_xyz)
    adj = build_adjacency(ref)
    frag1 = outer_terminal_fragment(adj, (LEFT[1], LEFT[2]), set(CENTER))
    frag4 = outer_terminal_fragment(adj, (RIGHT[1], RIGHT[2]), set(CENTER))
    center_all = sorted(set(range(len(ref))) - set(frag1) - set(frag4))
    frag2, frag3 = split_fragment_at_bond(adj, (CENTER[1], CENTER[2]), center_all)
    fragments = {1: frag1, 2: frag2, 3: frag3, 4: frag4}
    frag_of = {a: k for k, atoms in fragments.items() for a in atoms}

    fig_all, axes_all = plt.subplots(len(STRUCTURES), 2, figsize=(11, 4.6 * len(STRUCTURES)))

    for row, name in enumerate(STRUCTURES):
        atoms = read(sweep / name / "geometry.xyz")
        pos = atoms.positions
        bonds = [(i, j) for i in range(len(atoms)) for j in build_adjacency(atoms)[i] if i < j]

        f_axis = read_efield(sweep / name / "broken_symmetry" / "axis_p0.001" / "broken_symmetry.inp")
        f_perp = read_efield(sweep / name / "broken_symmetry" / "perp_p0.001" / "broken_symmetry.inp")
        f_axis_hs = read_efield(sweep / name / "high_spin" / "axis_p0.001" / "high_spin.inp")
        assert np.allclose(f_axis, f_axis_hs), f"{name}: HS and BS axis fields differ"
        u_axis, u_perp = f_axis / np.linalg.norm(f_axis), f_perp / np.linalg.norm(f_perp)

        c1, c4 = pos[frag1].mean(axis=0), pos[frag4].mean(axis=0)
        c23 = pos[frag2 + frag3].mean(axis=0)
        d14 = np.linalg.norm(c4 - c1)
        tip = c1 + d14 * u_axis
        miss = np.linalg.norm(tip - c4)
        angle = np.degrees(np.arccos(np.clip(u_axis @ ((c4 - c1) / d14), -1, 1)))
        perp_dot = u_axis @ u_perp
        print(f"{name:16s} |F|={np.linalg.norm(f_axis):.6f} a.u.  angle(F, c1->c4)={angle:.4f} deg  "
              f"tip miss={miss:.2e} A  axis.perp={perp_dot:.1e}")

        # principal axes of the molecule (view frame independent of the field vectors)
        X = pos - pos.mean(axis=0)
        _, vecs = np.linalg.eigh(X.T @ X)
        pc = vecs[:, ::-1]
        views = [("top view (PC1–PC2)", pc[:, 0], pc[:, 1]), ("side view (PC1–PC3)", pc[:, 0], pc[:, 2])]

        fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
        for ax_pair in (axs, axes_all[row]):
            for ax, (vtitle, ex, ey) in zip(ax_pair, views):
                P = lambda v: np.array([v @ ex, v @ ey])  # noqa: E731
                P2 = pos @ np.column_stack([ex, ey])
                for i, j in bonds:
                    same = frag_of[i] == frag_of[j]
                    ax.plot(*P2[[i, j]].T, color=FRAG_COLORS[frag_of[i]] if same else "#b5b4ae",
                            lw=1.6, solid_capstyle="round", zorder=1)
                heavy = np.array([s != "H" for s in atoms.get_chemical_symbols()])
                for k, idx in fragments.items():
                    idx = np.array(idx)
                    h = idx[heavy[idx]]
                    ax.scatter(*P2[h].T, s=22, color=FRAG_COLORS[k], edgecolor="white", lw=0.8, zorder=2,
                               label=f"fragment {k}")
                # centroids
                for c, lab in ((c1, "1"), (c4, "4")):
                    ax.scatter(*P(c), s=110, facecolor="white", edgecolor=INK, lw=1.4, zorder=4)
                    ax.annotate(lab, P(c), ha="center", va="center", fontsize=8, fontweight="bold", zorder=5)
                # field along axis: from c1, length d14
                ax.annotate("", xy=P(tip), xytext=P(c1), zorder=3,
                            arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2, mutation_scale=16, shrinkA=0, shrinkB=6))
                mid = 0.5 * (c1 + c4)
                ax.annotate(r"$\mathbf{F}_{\rm axis}$ (+)", P(mid), textcoords="offset points", xytext=(6, -12),
                            fontsize=9, color=INK)
                # perpendicular direction, dashed, from the 1-4 midpoint
                L = np.linalg.norm((c23 - mid) - ((c23 - mid) @ u_axis) * u_axis)
                ax.annotate("", xy=P(mid + L * u_perp), xytext=P(mid), zorder=3,
                            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4, ls="--", mutation_scale=12, shrinkA=0))
                ax.annotate(r"$\mathbf{F}_{\perp}$ (+)", P(mid + L * u_perp), textcoords="offset points",
                            xytext=(4, 4), fontsize=9, color=MUTED)
                ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.set_title(f"{name} — {vtitle}", fontsize=10, color=INK, loc="left")
            ax_pair[0].legend(loc="lower left", fontsize=8, frameon=False, ncol=4,
                              bbox_to_anchor=(0, -0.12), handletextpad=0.2, columnspacing=0.8)
            ax_pair[1].text(1.0, -0.08,
                            f"EField from axis_p0.001 input: ({f_axis[0]:+.6f}, {f_axis[1]:+.6f}, {f_axis[2]:+.6f}) a.u.\n"
                            f"angle to centroid 1→4 = {angle:.3f}°, tip-to-centroid-4 = {miss:.1e} Å",
                            transform=ax_pair[1].transAxes, ha="right", va="top", fontsize=8, color=MUTED)
        fig.tight_layout()
        fig.savefig(out_dir / f"field_axis_{name}.png", dpi=200)
        plt.close(fig)

    fig_all.tight_layout()
    fig_all.savefig(out_dir / "field_axis_all.png", dpi=150)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
