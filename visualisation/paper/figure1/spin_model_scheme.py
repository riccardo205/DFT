#!/usr/bin/env python3
"""
Schematic four-spin model of two stacked molecules (companion of molecule_stack_view.py).

Each molecule is a thin disc (a plane seen in perspective) with one spin at each end, drawn as an arrow sticking
out of the plane (red up, blue down, as the radical arrows of the stack figure). Sites 1, 2 on the lower molecule,
3, 4 on the upper one (shifted by SHIFT_X like the stack figure). All six exchange couplings are drawn between the
sites as curved double arrows: J12 and J34 arch out of each molecular plane, J13, J14, J23, J24 bow between the
molecules (curvature and side per coupling in J_RAD), each with its label.

Output: OUT_DIR/spin_model_scheme.pdf / .png at exactly SIZE_CM (Arial, transparent). Settings in the __main__ block.
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import visualisation.paper.figure1.molecule_torsion_view as V  # noqa: E402


def slab(ax, c, w, h, thick, face, side, edge, lw, z):
    """Molecule as a thin disc seen in perspective: side band (thickness) + top face."""
    ax.add_patch(Ellipse(c - [0, thick], w, h, facecolor=side, edgecolor=edge, lw=lw, zorder=z))
    ax.add_patch(Rectangle(c - [w / 2, thick], w, thick, facecolor=side, edgecolor="none", zorder=z))
    for sx in (-1, 1):
        ax.plot([c[0] + sx * w / 2] * 2, [c[1] - thick, c[1]], color=edge, lw=lw, zorder=z, solid_capstyle="butt")
    ax.add_patch(Ellipse(c, w, h, facecolor=face, edgecolor=edge, lw=lw, zorder=z + 0.1))


def outline(c, A, B, persp, phi, n=360):
    """Molecule as a flat ellipse in its plane (semi-axes A along the molecular axis, B across), turned by phi
    (rad, anticlockwise seen from above) and seen in perspective: depth (in-plane y) squeezed by persp."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = A * np.cos(t), B * np.sin(t)
    xr, yr = x * np.cos(phi) - y * np.sin(phi), x * np.sin(phi) + y * np.cos(phi)
    return c + np.column_stack([xr, persp * yr])


def slab_turned(ax, top, thick, face, side, edge, lw, z):
    """Thin slab whose top face is the polygon `top`: side band = hull of top and top shifted down by thick."""
    both = np.vstack([top, top - [0, thick]])
    hull = both[ConvexHull(both).vertices]
    ax.add_patch(Polygon(hull, closed=True, facecolor=side, edgecolor=edge, lw=lw, zorder=z, joinstyle="round"))
    ax.add_patch(Polygon(top, closed=True, facecolor=face, edgecolor=edge, lw=lw, zorder=z + 0.1, joinstyle="round"))


def arc_point(a, b, rad, t=0.5):
    """Point at parameter t of matplotlib's arc3 connection (quadratic Bezier a -> control -> b) and the unit
    normal on its convex side."""
    d = b - a
    c = 0.5 * (a + b) + rad * np.array([d[1], -d[0]])
    p = (1 - t) ** 2 * a + 2 * t * (1 - t) * c + t ** 2 * b
    tan = 2 * (1 - t) * (c - a) + 2 * t * (b - c)
    n = np.array([-tan[1], tan[0]])
    n = n / (np.linalg.norm(n) or 1)
    if n @ (c - 0.5 * (a + b)) < 0:
        n = -n
    return p, n


def main(out_dir, size_cm, font, label_size, transparent, ell_w, ell_h, thick, site_frac, sep, shift_x, spins,
         arrow_len, arrow_lw, arrow_head, col_up, col_down, mol_face, mol_side, mol_edge, mol_lw, j_col, j_lw,
         j_head, j_style, j_rad, label_t, label_out, label_nudge, pad, pairs=None, out_name="spin_model_scheme", texts=None,
         azimuth=(0.0, 0.0), mol_aspect=None):
    """pairs: couplings to draw, e.g. ("12", "34", "13") (None: all six). texts: label text per coupling
    (default J_ij). azimuth: (lower, upper) molecule turned about the vertical axis (deg, anticlockwise seen from
    above), spins kept at the ends of the molecular axis. mol_aspect: in-plane width / length of the molecule
    (None: round disc); the perspective is then set so that the unturned molecule is ell_w x ell_h on screen."""
    V.set_fonts(font)
    centres = {1: np.array([0.0, 0.0]), 2: np.array([shift_x, sep])}           # lower, upper molecule
    # molecule k: flat ellipse in its plane (A along the molecular axis, B across) seen with perspective persp;
    # the spin sites sit on the molecular axis at +-site_frac A and turn with the molecule
    A = ell_w / 2
    B = A if mol_aspect is None else A * mol_aspect
    persp = (ell_h / 2) / B
    plain = mol_aspect is None and not any(azimuth)             # original round discs (figure 1)
    tops, site = {}, {}
    for k, (left, right) in ((1, (1, 2)), (2, (3, 4))):
        phi = np.radians(azimuth[k - 1])
        tops[k] = outline(centres[k], A, B, persp, phi)
        d = site_frac * A * np.array([np.cos(phi), persp * np.sin(phi)])
        site[left], site[right] = centres[k] - d, centres[k] + d

    fig, ax = V.new_figure(size_cm)
    for k in (1, 2):
        if plain:
            slab(ax, centres[k], ell_w, ell_h, thick, mol_face[k - 1], mol_side[k - 1], mol_edge[k - 1], mol_lw, 1)
        else:
            slab_turned(ax, tops[k], thick, mol_face[k - 1], mol_side[k - 1], mol_edge[k - 1], mol_lw, 1)

    pts = [site[i] for i in site]
    labels, spin_arrows = [], []
    for p, (a, b) in {"12": (1, 2), "34": (3, 4), "13": (1, 3), "24": (2, 4), "14": (1, 4), "23": (2, 3)}.items():
        if pairs is not None and p not in pairs:
            continue
        intra = p in ("12", "34")
        col = j_col["intra" if intra else "inter"]
        ax.add_patch(FancyArrowPatch(site[a], site[b], arrowstyle=j_style, mutation_scale=j_head, lw=j_lw, color=col,
                                     connectionstyle=f"arc3,rad={j_rad[p]}", shrinkA=3, shrinkB=3,
                                     zorder=3 if intra else 2, capstyle="round"))
        m, out = arc_point(site[a], site[b], j_rad[p], label_t.get(p, 0.5))
        # label offset in points from the arc on its convex side, anchored on the side facing the arc, so the
        # text box never covers the arc whatever its direction
        ha = "left" if out[0] > 0.35 else "right" if out[0] < -0.35 else "center"
        va = "bottom" if out[1] > 0.35 else "top" if out[1] < -0.35 else "center"
        off = label_out * out + np.array(label_nudge.get(p, (0, 0)))
        labels.append(ax.annotate((texts or {}).get(p, rf"$J_{{{p}}}$"), m, xytext=off, textcoords="offset points", fontsize=label_size,
                                  color=col, ha=ha, va=va, zorder=6))
        pts.append(m)

    for i, sgn in spins.items():                                               # spins out of the plane
        col = col_up if sgn > 0 else col_down
        if sgn > 0:
            tip = site[i] + [0, arrow_len]
        else:                                    # same visible length below the disc rim as the up arrows above
            k = 1 if i in (1, 2) else 2
            c = centres[k]
            if plain:
                rim = c[1] - thick - ell_h / 2 * np.sqrt(max(0.0, 1 - ((site[i][0] - c[0]) / (ell_w / 2)) ** 2))
            else:                                # lower edge of the slab under the site
                bot = tops[k] - [0, thick]
                near = bot[np.abs(bot[:, 0] - site[i][0]) < 0.02 * ell_w]
                rim = near[:, 1].min() if len(near) else c[1] - thick - ell_h / 2
            tip = np.array([site[i][0], rim - arrow_len])
        # down spins point through the disc: drawn under it, so they come out of its lower side
        spin_arrows.append(ax.add_patch(FancyArrowPatch(site[i], tip, arrowstyle="-|>", mutation_scale=arrow_head,
                                                        lw=arrow_lw, color=col, shrinkA=0, shrinkB=0,
                                                        zorder=5 if sgn > 0 else 0.5, capstyle="round")))
        ax.scatter(*site[i], s=14, color=col, edgecolor="white", lw=0.4, zorder=5.5)
        pts.append(tip)

    pts = np.vstack(pts + [c + [sx * ell_w / 2, sy * ell_h / 2 - (thick if sy < 0 else 0)]
                           for c in centres.values() for sx in (-1, 1) for sy in (-1, 1)] if plain else
                    pts + [tops[k] for k in tops] + [tops[k] - [0, thick] for k in tops])
    lo, hi = pts.min(axis=0), pts.max(axis=0)

    def apply_limits(lo, hi):
        """Limits = content + pad, widened in x or y to the aspect of the axes box, so that the equal-aspect
        adjustment has nothing left to change (with 'datalim' it may otherwise shrink one axis and cut content)."""
        x0, x1, y0, y1 = lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad
        bb = ax.get_position().transformed(fig.transFigure)
        target = bb.width / bb.height
        if (x1 - x0) / (y1 - y0) < target:
            d = 0.5 * ((y1 - y0) * target - (x1 - x0)); x0, x1 = x0 - d, x1 + d
        else:
            d = 0.5 * ((x1 - x0) / target - (y1 - y0)); y0, y1 = y0 - d, y1 + d
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

    for it in range(30):                     # grow the limits until the labels (sized in points) fit inside
        apply_limits(lo, hi)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        boxes = [t.get_window_extent(r) for t in labels]
        inv = ax.transData.inverted()
        corners = np.vstack([inv.transform([[b.x0, b.y0], [b.x1, b.y1]]) for b in boxes])
        new_lo, new_hi = np.minimum(lo, corners.min(axis=0)), np.maximum(hi, corners.max(axis=0))
        if np.allclose(new_lo, lo, atol=1e-3) and np.allclose(new_hi, hi, atol=1e-3):
            break
        lo, hi = new_lo, new_hi
    else:
        print("  [!] label fitting did not converge: the labels may touch the edge")
    apply_limits(lo, hi)
    for t, b in zip(labels, boxes):          # report labels touching a spin arrow
        if any(b.overlaps(a.get_window_extent(r)) for a in spin_arrows):
            print(f"  [!] {t.get_text()} overlaps a spin arrow: adjust LABEL_T / LABEL_NUDGE")
    V.save(fig, out_dir / out_name, transparent)


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    OUT_DIR = HERE / "molecule_torsions"
    SIZE_CM = (3.2, 3.2)
    FONT = "Arial"
    LABEL_SIZE = 7                              # pt, J labels
    TRANSPARENT = True
    PAD = 0.15                                  # margin (drawing units)

    # geometry (drawing units; ellipse width = 1 molecule)
    ELL_W, ELL_H = 4.0, 1.0                     # ellipse axes (height < width: plane seen in perspective)
    THICK = 0.12                                # thickness of the molecule discs
    SITE_FRAC = 0.72                            # spin sites at +-SITE_FRAC * half width from the centre
    SEP = 2.6                                   # vertical distance between the two molecules
    SHIFT_X = 0.6                               # horizontal shift of the upper molecule (as in the stack figure)

    SPINS = {1: +1, 2: -1, 3: +1, 4: -1}        # +1 up (red), -1 down (blue); 1, 2 lower molecule, 3, 4 upper
    ARROW_LEN = 0.85
    ARROW_LW = 1.1
    ARROW_HEAD = 5
    COL_UP, COL_DOWN = "#C0392B", "#1F4E9A"

    MOL_FACE = ("#E8EEF4", "#E8EEF4")           # top face of the lower / upper molecule disc
    MOL_SIDE = ("#C5D0DB", "#C5D0DB")           # side band (thickness)
    MOL_EDGE = ("#8C9BAA", "#8C9BAA")
    MOL_LW = 0.4

    J_COL = {"intra": "black", "inter": "black"}  # J12, J34 / intermolecular couplings (arrows and labels)
    J_LW = 0.55
    J_HEAD = 3
    J_STYLE = "<|-|>"                           # "-" for plain curves
    # bending of each coupling (arc3 rad: sign = side, size = curvature). Intra arcs rise out of the plane,
    # inter arcs bow outwards; the two diagonals bend opposite ways so they do not cross at one point.
    J_RAD = {"12": -0.4, "34": -0.4, "13": -0.45, "24": 0.45, "14": -0.15, "23": 0.15}
    LABEL_T = {"12": 0.5, "34": 0.5, "13": 0.5, "24": 0.5, "14": 0.3, "23": 0.3}   # label position along each arc
    LABEL_OUT = 1.5                             # pt between the arc and the nearest edge of its label (convex side)
    LABEL_NUDGE = {"12": (0, 0), "34": (0, 0), "13": (0, 0), "24": (0, 0), "14": (0, 0), "23": (0, 0)}

    main(OUT_DIR, SIZE_CM, FONT, LABEL_SIZE, TRANSPARENT, ELL_W, ELL_H, THICK, SITE_FRAC, SEP, SHIFT_X, SPINS,
         ARROW_LEN, ARROW_LW, ARROW_HEAD, COL_UP, COL_DOWN, MOL_FACE, MOL_SIDE, MOL_EDGE, MOL_LW, J_COL, J_LW, J_HEAD,
         J_STYLE, J_RAD, LABEL_T, LABEL_OUT, LABEL_NUDGE, PAD)
