#!/usr/bin/env python3
"""
Show which atoms each FlipSpin configuration of a dimer flips.

Top: atom-index map. One row per configuration, one column per atom (0 .. 2N-1); every atom
is coloured by its radical half (A-D, from radical_fragments.json); flipped atoms are solid,
unflipped atoms faint. The strip above marks C/H and the monomer boundary.
Bottom: the dimer projected on its two main axes, one panel per configuration, in the order
uuuu -> single flips -> pair flips; flipped atoms solid, the rest faint.

Usage: python3 plot_flip_atoms.py <run>_broken_symmetry/<stage>/<dimer> [more dimer dirs]
Writes <dimer>/flip_atoms_map.png.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs
from matplotlib.patches import Patch

COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDER = ["uuuu", "duuu", "uduu", "uudu", "uuud", "dduu", "dudu", "duud"]
FLIP_RE = re.compile(r"^\s*FlipSpin\s+(.*)$", re.M)


def flip_list(inp):
    m = FLIP_RE.search(inp.read_text())
    return {int(x) for x in m.group(1).replace(",", " ").split()} if m else set()


def plot_dimer(dimer_dir):
    dimer_dir = Path(dimer_dir)
    atoms = read(dimer_dir / "uuuu" / "geometry.xyz")
    sym = np.array(atoms.get_chemical_symbols())
    n_tot, n = len(atoms), len(atoms) // 2
    frags = json.loads((dimer_dir / "radical_fragments.json").read_text())["fragments"]
    owner = np.empty(n_tot, dtype="<U1")
    for lab, idx in frags.items():
        owner[idx] = lab
    configs = [c for c in ORDER if (dimer_dir / c / f"{c}.inp").exists()]
    flips = {c: flip_list(dimer_dir / c / f"{c}.inp") for c in configs}

    fig = plt.figure(figsize=(16, 4.2 + 2.9 * 2))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.35, 1, 1], hspace=0.6, wspace=0.05)

    # ---- index map
    ax = fig.add_subplot(gs[0, :])
    for r, c in enumerate(configs):
        y = len(configs) - 1 - r
        for i in range(n_tot):
            on = i in flips[c]
            ax.add_patch(plt.Rectangle((i, y + 0.08), 1, 0.84, color=COLORS[owner[i]], alpha=1.0 if on else 0.18,
                                       linewidth=0))
        ax.text(-1.5, y + 0.5, f"{c}  ({len(flips[c])} flipped)", ha="right", va="center", fontsize=9, color=INK)
    top = len(configs)
    for i in range(n_tot):  # element strip
        ax.add_patch(plt.Rectangle((i, top + 0.15), 1, 0.6, color=INK if sym[i] == "C" else "#b5b4ae", linewidth=0))
    ax.text(-1.5, top + 0.45, "element (dark C, light H)", ha="right", va="center", fontsize=8.5, color=INK_2)
    ax.axvline(n, color=INK, linewidth=1.5)
    ax.text(n / 2, top + 1.05, f"monomer 1: atoms 0–{n - 1}", ha="center", fontsize=10, color=INK)
    ax.text(n + n / 2, top + 1.05, f"monomer 2: atoms {n}–{n_tot - 1}", ha="center", fontsize=10, color=INK)
    ax.set_xlim(0, n_tot)
    ax.set_ylim(0, top + 1.5)
    ax.set_yticks([])
    ax.set_xticks(range(0, n_tot + 1, 10))
    ax.set_xlabel("ORCA atom index (0-based, as in FlipSpin)", color=INK_2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    handles = [Patch(color=COLORS[k], label=f"radical {k} ({len(v)} atoms, monomer {1 if max(v) < n else 2})")
               for k, v in frags.items()]
    handles += [Patch(color="#999999", alpha=1.0, label="solid = flipped"),
                Patch(color="#999999", alpha=0.18, label="faint = not flipped")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=6, frameon=False, fontsize=8.5)
    ax.set_title(f"{dimer_dir.parent.parent.name}/{dimer_dir.parent.name}/{dimer_dir.name}: FlipSpin atoms per configuration",
                 color=INK, fontsize=11, loc="left")

    # ---- structure panels
    pos = atoms.positions - atoms.positions.mean(0)
    axes2 = np.linalg.svd(pos, full_matrices=False)[2][:2]
    xy = pos @ axes2.T
    nl = NeighborList(natural_cutoffs(atoms), self_interaction=False, bothways=False)
    nl.update(atoms)
    bonds = [(i, j) for i in range(n_tot) for j in nl.get_neighbors(i)[0]]
    panel_configs = [c for c in ["uuuu", "duuu", "uduu", "uudu", "uuud", "dduu", "dudu", "duud"] if c in configs][:8]
    for k, c in enumerate(panel_configs):
        ax = fig.add_subplot(gs[1 + k // 4, k % 4])
        for i, j in bonds:
            ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=GRID, linewidth=0.8, zorder=1)
        on = np.array([i in flips[c] for i in range(n_tot)])
        cols = [COLORS[o] for o in owner]
        size = np.where(sym == "C", 26, 9)
        ax.scatter(xy[~on, 0], xy[~on, 1], s=size[~on], c=[cols[i] for i in np.where(~on)[0]], alpha=0.2, zorder=2,
                   linewidths=0)
        ax.scatter(xy[on, 0], xy[on, 1], s=size[on] * 1.4, c=[cols[i] for i in np.where(on)[0]], zorder=3,
                   edgecolors=INK, linewidths=0.4)
        for lab, idx in frags.items():
            cx, cy = xy[idx].mean(0)
            state = "↓" if (set(idx) & flips[c]) else "↑"
            ax.text(cx, cy, f"{lab}{state}", fontsize=11, fontweight="bold", color=INK, ha="center", va="center",
                    zorder=4, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=COLORS[lab], alpha=0.85))
        ax.set_title(f"{c}: {len(flips[c])} atoms flipped", fontsize=9.5, color=INK)
        ax.set_aspect("equal")
        ax.axis("off")
    out = dimer_dir / "flip_atoms_map.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(f"wrote {plot_dimer(d)}")
