#!/usr/bin/env python3
"""
Simple view of one diradical molecule (real 3D geometry, orthographic projection) showing the structure and the
three torsional degrees of freedom theta_L, theta_C, theta_R.

  carbons only        hydrogens are not drawn; bonds thin black lines
  atom colour         spin sublattice: red = A, blue = B. Taken from the sign of the Mulliken spin of the
                      high-spin (triplet) output next to the geometry, flipped on the right half of the molecule
                      (the half of the right radical, split at the central bond): this is the alternation of the
                      open-shell singlet, where the left radical is up and the right radical down. (The fluorenyl
                      5-ring is odd, so a pure graph 2-colouring does not exist.) Without an output file the
                      script falls back to a 2-colouring of the carbon graph (BFS, odd rings resolved arbitrarily).
  radical arrows      at the two radical carbons C9: arrow up in the colour of its sublattice (red, A) or down
                      (blue, B)
  torsions            elliptical arrow around each rotating bond (rotation about the bond axis) and
                      the label theta_L / theta_C / theta_R (optionally with the value of the dihedral)
Orientation: z (towards the viewer) = average normal of the two central anthracene planes, x = longest
direction of the molecule in that plane (left radical on the left); then the extra rotations EXTRA_ROT
(degrees, about the screen x, y, z axes, applied in that order).

Output: OUT_DIR/<geometry folder name>_torsions.pdf / .png, saved at exactly SIZE_CM (transparent background if
TRANSPARENT), legend below the molecule. Settings in the __main__ block.
"""

import re
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath

MULLIKEN_RE = re.compile(r"MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS\s*\n-+\n(.*?)\nSum of atomic", re.S)
ATOM_RE = re.compile(r"^\s*(\d+)\s+\S+\s*:\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)", re.M)


# =========================
# GEOMETRY
# =========================
def adjacency(atoms, cc_max=1.75):
    """C-C bonds only (hydrogens are not drawn; a bay H close to the other unit would otherwise close a ring)."""
    X = atoms.get_positions()
    C = [i for i, a in enumerate(atoms) if a.symbol == "C"]
    adj = {i: set() for i in range(len(atoms))}
    for k, i in enumerate(C):
        for j in C[k + 1:]:
            if np.linalg.norm(X[i] - X[j]) < cc_max:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def side_of_bond(adj, start, bond, keep=None):
    """Atoms reachable from `start` without crossing `bond` (optionally restricted to `keep`)."""
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if {u, v} == set(bond) or v in seen or (keep is not None and v not in keep):
                continue
            seen.add(v)
            stack.append(v)
    return seen


def plane_normal(P):
    P = P - P.mean(axis=0)
    return np.linalg.svd(P)[2][2]


def rot(axis, deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def orient(X, carbons, anth1, anth2, left_rad, right_rad, extra):
    """Rotation: z = mean normal of the two anthracene planes, x = longest in-plane direction."""
    n1, n2 = plane_normal(X[list(anth1)]), plane_normal(X[list(anth2)])
    z = n1 + (n2 if n1 @ n2 > 0 else -n2)
    z /= np.linalg.norm(z)
    C = X[carbons] - X[carbons].mean(axis=0)
    C2 = C - np.outer(C @ z, z)                               # projection on the plane
    x = np.linalg.svd(C2)[2][0]
    x -= (x @ z) * z
    x /= np.linalg.norm(x)
    if (X[right_rad] - X[left_rad]) @ x < 0:
        x = -x
    R = np.array([x, np.cross(z, x), z])                      # rows: screen axes
    for ax, deg in zip("xyz", extra):
        R = rot(ax, deg) @ R
    return (X - X[carbons].mean(axis=0)) @ R.T


def mulliken_spins(out_file, n):
    blocks = MULLIKEN_RE.findall(out_file.read_text(errors="ignore"))
    if not blocks:
        return None
    rho = np.zeros(n)
    for m in ATOM_RE.finditer(blocks[-1]):
        rho[int(m.group(1))] = float(m.group(3))
    return rho


def bfs_colouring(adj, carbons, start):
    col, q = {start: 1}, deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v in carbons and v not in col:
                col[v] = -col[u]
                q.append(v)
    return col


# =========================
# DRAWING
# =========================
def torsion_arc(ax, Y, b, c, radius, span_deg, min_ratio, colour, lw, zorder, head=11):
    """Circular arrow around the bond b-c, drawn as an ellipse centred on the bond midpoint: major axis
    perpendicular to the projected bond, minor axis along it (a rotation about an axis seen at an angle; the
    flatter the bond lies in the screen plane, the flatter the ellipse, but at least min_ratio)."""
    u3 = Y[c] - Y[b]
    u3 /= np.linalg.norm(u3)
    u = u3[:2] / (np.linalg.norm(u3[:2]) or 1.0)             # projected bond direction
    v = np.array([-u[1], u[0]])
    ratio = max(min_ratio, abs(u3[2]))
    mid = 0.5 * (Y[b, :2] + Y[c, :2])
    t = np.radians(np.linspace(0, span_deg, 120)) + np.radians((360 - span_deg) / 2) - np.pi / 2
    pts = mid + radius * (np.outer(np.cos(t), v) + ratio * np.outer(np.sin(t), u))
    ax.add_patch(FancyArrowPatch(path=MplPath(pts), arrowstyle="-|>", mutation_scale=head, color=colour, lw=lw,
                                 zorder=zorder, shrinkA=0, shrinkB=0))
    return mid, v


def set_fonts(font):
    plt.rcParams.update({"font.family": font, "mathtext.fontset": "custom", "mathtext.rm": font,
                         "mathtext.it": f"{font}:italic", "mathtext.bf": f"{font}:bold"})


def prepare(geometry, out_file, left, centre, right, radicals, adj=None):
    """Geometry, C-C graph, halves, anthracene units, radicals and sublattices (+1 A, -1 B) of the molecule.
    adj: bond graph to use instead of the distance-based one (e.g. that of a reference molecule with the same
    numbering, when this geometry has a short non-bonded C-C contact)."""
    atoms = read(geometry)
    adj = adjacency(atoms) if adj is None else adj
    carbons = [i for i, a in enumerate(atoms) if a.symbol == "C"]
    cset = set(carbons)
    half_left = side_of_bond(adj, centre[1], (centre[1], centre[2]))
    outer_left = side_of_bond(adj, left[1], (left[1], left[2]))           # side of the left radical
    outer_right = side_of_bond(adj, right[2], (right[1], right[2]))
    left_rad = next(r for r in radicals if r in half_left)
    right_rad = next(r for r in radicals if r not in half_left)
    rho = mulliken_spins(out_file, len(atoms)) if out_file and Path(out_file).exists() else None
    if rho is not None:
        sub = {i: (1 if rho[i] >= 0 else -1) * (1 if i in half_left else -1) for i in carbons}
        src = f"Mulliken spins of {Path(out_file).name}"
    else:
        sub = bfs_colouring(adj, cset, left_rad)
        src = "graph 2-colouring (no output file)"
    return {"atoms": atoms, "X": atoms.get_positions(), "adj": adj, "carbons": carbons,
            "anth1": (half_left - outer_left) & cset, "anth2": cset - half_left - outer_right,
            "left_rad": left_rad, "right_rad": right_rad, "sub": sub, "src": src,
            "torsions": {"L": left, "C": centre, "R": right}}


def oriented(m, extra_rot):
    return orient(m["X"], m["carbons"], m["anth1"], m["anth2"], m["left_rad"], m["right_rad"], extra_rot)


def arrow_tips(m, Y, arrow_len):
    return {r: Y[r, :2] + np.array([0, arrow_len if m["sub"][r] > 0 else -arrow_len])
            for r in (m["left_rad"], m["right_rad"])}


def draw_molecule(ax, m, Y, st):
    """Bonds (thin black), atoms coloured by sublattice, spin arrows at the radicals. Depth order: every bond
    lies just behind its farther atom, atoms in front by their depth. Optional depth cues (style keys):
    depth_fade (0..1) mixes the farthest atoms/bonds with white, depth_scale (0..1) shrinks the farthest atoms,
    arrow_depth (default True) draws the spin arrows at the depth of their radical (False: always on top)."""
    C, adj, sub = m["carbons"], m["adj"], m["sub"]
    z0, zr = Y[C, 2].min(), (np.ptp(Y[C, 2]) or 1.0)
    depth = lambda z: 2 + 2 * (z - z0) / zr
    far = lambda z: 1 - (z - z0) / zr                       # 0 front, 1 back
    fade, shrink = st.get("depth_fade", 0.0), st.get("depth_scale", 0.0)
    mix = lambda col, z: tuple((1 - fade * far(z)) * np.array(to_rgb(col)) + fade * far(z))
    for i in C:
        for j in adj[i]:
            if j > i:
                z = min(Y[i, 2], Y[j, 2])
                ax.plot(*zip(Y[i, :2], Y[j, :2]), color=mix("black", 0.5 * (Y[i, 2] + Y[j, 2])),
                        lw=st["bond_lw"], solid_capstyle="round", zorder=depth(z) - 1e-3)
    for i in C:
        z = Y[i, 2]
        ax.scatter(*Y[i, :2], s=st["atom_size"] * (1 - shrink * far(z)),
                   color=mix(st["col_a"] if sub[i] > 0 else st["col_b"], z), edgecolor=mix("black", z),
                   lw=st["atom_edge_lw"], zorder=depth(z))
    for r, tip in arrow_tips(m, Y, st["arrow_len"]).items():
        ax.add_patch(FancyArrowPatch(Y[r, :2], tip, arrowstyle="-|>", mutation_scale=st["arrow_head"],
                                     lw=st["arrow_lw"], color=st["col_a"] if sub[r] > 0 else st["col_b"],
                                     # at the depth of its radical: atoms and bonds in front of it cover it
                                     zorder=depth(Y[r, 2]) + 1e-3 if st.get("arrow_depth", True) else 10,
                                     shrinkA=0, shrinkB=0))


def new_figure(size_cm, ax_box=(0, 0, 1, 1)):
    fig = plt.figure(figsize=(size_cm[0] / 2.54, size_cm[1] / 2.54))
    ax = fig.add_axes(ax_box)
    ax.set_aspect("equal", adjustable="datalim")             # fills the box, keeps 1 Å = 1 Å
    ax.axis("off")
    return fig, ax


def save(fig, stem, transparent, dpi=600):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=dpi, transparent=transparent)   # exact size_cm: no bbox_inches="tight"
    plt.close(fig)
    print(f"wrote {stem}.pdf/.png")


def main(geometry, out_file, out_dir, left, centre, right, radicals, extra_rot, size_cm, font, legend_size,
         label_size, legend_height, transparent, style, arc_radius, arc_span, arc_ratio, arc_lw, arc_head, arc_colour,
         label_dist, label_offsets, show_values, pad, ghost=None):
    """ghost: dict(alpha=...) draws a faded up arrow (colour of sublattice A) at the down-spin radical, to show
    that it can also point up (singlet / triplet); None: no extra arrow."""
    set_fonts(font)
    m = prepare(geometry, out_file, left, centre, right, radicals)
    values = {k: m["atoms"].get_dihedral(*q) for k, q in m["torsions"].items()}
    Y = oriented(m, extra_rot)

    fig, ax = new_figure(size_cm, (0, legend_height, 1, 1 - legend_height))
    draw_molecule(ax, m, Y, style)
    centroid = Y[m["carbons"], :2].mean(axis=0)
    pts = [Y[m["carbons"], :2], np.array(list(arrow_tips(m, Y, style["arrow_len"]).values()))]
    if ghost:                                                     # faded up arrow at the down-spin radical
        for r in (m["left_rad"], m["right_rad"]):
            if m["sub"][r] < 0:
                base = Y[r, :2] + np.array([ghost.get("dx", 0.0), 0.0])   # sideways shift (Å) off the bond
                tip = base + np.array([0, style["arrow_len"]])
                ax.add_patch(FancyArrowPatch(base, tip, arrowstyle="-|>", mutation_scale=style["arrow_head"],
                                             lw=style["arrow_lw"], color=style["col_a"], alpha=ghost["alpha"],
                                             shrinkA=0, shrinkB=0, zorder=10))
                pts.append(tip[None])
    for k, (a, b, c, d) in m["torsions"].items():
        mid, v = torsion_arc(ax, Y, b, c, arc_radius, arc_span, arc_ratio, arc_colour, arc_lw, 9, arc_head)
        rad = [r for r in (m["left_rad"], m["right_rad"]) if r in (b, c)]
        if rad:                                                   # side opposite to that radical's spin arrow
            away = v if v[1] * m["sub"][rad[0]] <= 0 else -v
        else:                                                     # perpendicular to the bond, away from the centre
            away = v if v @ (mid - centroid) >= 0 else -v
        pos = mid + (arc_radius + label_dist) * away + np.array(label_offsets.get(k, (0, 0)))
        ax.text(*pos, rf"$\theta_{{{k}}}$" + (f" = {abs(values[k]):.0f}°" if show_values else ""),
                fontsize=label_size, ha="center", va="center", color=arc_colour, zorder=11)
        pts.append(pos[None])
    pts = np.vstack(pts)
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ms = np.sqrt(style["atom_size"]) * 1.3
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=ms, mfc=style[c], mec="black", mew=style["atom_edge_lw"],
                               label=f"sublattice {s}") for c, s in (("col_a", "A"), ("col_b", "B"))],
               loc="lower center", ncol=2, frameon=False, fontsize=legend_size, handletextpad=0.2,
               columnspacing=1.0, borderaxespad=0.1)
    save(fig, out_dir / f"{geometry.parent.name}_torsions", transparent)
    print(f"sublattices from {m['src']}; dihedrals: " + ", ".join(f"theta_{k} = {v:.1f}" for k, v in values.items()))


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    REPO = Path(__file__).resolve().parents[3]         # DFT root (script in visualisation/paper/figure1)
    SCAN = REPO / "runs" / "orca" / "short-2OS" / "angle_scans" / "angle_scan_relaxation" / "L045_C040_R045"
    GEOMETRY = SCAN / "geometry.xyz"
    OUT_FILE = SCAN / "high_spin.out"           # Mulliken spins for the sublattices (None: graph 2-colouring)
    OUT_DIR = Path(__file__).resolve().parent / "outputs"

    # 0-based atom indices (same as prepare_central_scan_approx_90.py / draw_torsion_scheme.py)
    LEFT = (42, 41, 0, 5)
    CENTRE = (4, 9, 17, 18)
    RIGHT = (21, 23, 28, 29)
    RADICALS = (41, 28)                         # radical carbons C9

    EXTRA_ROT = (0.0, 0.0, 0.0)                 # extra rotation (deg) about screen x, y, z, applied in this order

    # figure (saved at exactly this size)
    SIZE_CM = (5.0, 3.5)
    FONT = "Arial"
    LEGEND_SIZE = 7                             # pt
    LABEL_SIZE = 5                              # pt, theta_L / theta_C / theta_R
    LEGEND_HEIGHT = 0.10                        # fraction of the figure height kept for the legend at the bottom
    TRANSPARENT = True
    PAD = 0.4                                   # Å of empty margin around the drawing

    STYLE = {"atom_size": 5.0,                  # marker area (pt^2)
             "atom_edge_lw": 0.15,
             "bond_lw": 0.4,
             "arrow_len": 1.6,                  # Å, radical spin arrows
             "arrow_lw": 0.8,
             "arrow_head": 4,                   # arrow head size (pt)
             "col_a": "#C0392B", "col_b": "#1F4E9A"}   # sublattice A (red), B (blue)

    ARC_RADIUS = 0.6                            # Å, radius of the torsion arrows
    ARC_SPAN = 300                              # deg of the torsion arrow arc
    ARC_RATIO = 0.35                            # minimum minor/major axis ratio of the torsion arrow ellipse
    ARC_LW = 0.5
    ARC_HEAD = 3                                # torsion arrow head size (pt)
    ARC_COLOUR = "0.2"
    LABEL_DIST = 0.5                            # Å, label distance beyond the arrow (perpendicular to the bond)
    LABEL_OFFSETS = {"L": (0, 0), "C": (0, 0), "R": (0, 0)}   # Å, manual nudges of the labels
    SHOW_VALUES = False                         # True: "theta_L = 45°"
    GHOST = None                                # faded red up arrow at the blue (down) radical: it can point
                                                # either way (dx: Å sideways, off the C-C bond above it); None: off

    main(GEOMETRY, OUT_FILE, OUT_DIR, LEFT, CENTRE, RIGHT, RADICALS, EXTRA_ROT, SIZE_CM, FONT, LEGEND_SIZE,
         LABEL_SIZE, LEGEND_HEIGHT, TRANSPARENT, STYLE, ARC_RADIUS, ARC_SPAN, ARC_RATIO, ARC_LW, ARC_HEAD, ARC_COLOUR,
         LABEL_DIST, LABEL_OFFSETS, SHOW_VALUES, PAD, GHOST)
