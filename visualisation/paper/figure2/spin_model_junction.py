#!/usr/bin/env python3
"""
Schematic single-molecule junction in the simplified style of spin_model_scheme.py.

One molecule (thin disc seen in perspective) with a spin at each end (red up, blue down), contacted by two
electrodes drawn as gold triangles whose tips touch the two ends of the molecule. Concentric rings (alternately
coloured and empty from the outside in) around the left spin, extending over the tip of the left electrode,
symbolise its Kondo screening; small down spins around that tip
are the screening conduction electrons. Optionally an electrode tip is tinted (KONDO_LEAD).

Output: OUT_DIR/spin_model_junction.pdf / .png at exactly SIZE_CM (Arial, transparent). Settings in the __main__
block.
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, PathPatch, Polygon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                          # DFT root: makes visualisation.* importable
import visualisation.paper.figure1.molecule_torsion_view as V  # noqa: E402
from visualisation.paper.figure1.spin_model_scheme import arc_point, slab  # noqa: E402


def rgba_field(colour, alpha):
    """RGBA image of one colour with a per-pixel alpha."""
    img = np.zeros(alpha.shape + (4,))
    img[..., :3] = to_rgb(colour)
    img[..., 3] = np.clip(alpha, 0, 1)
    return img


def shaded(ax, X, Y, a, colour, alpha, level, halo, line_colour, line_lw, z, clip=None):
    """Field a (max ~1) drawn as a fill of `colour` inside the contour a = level (alpha from 0.75 alpha at the edge
    to alpha at the centre), a soft halo outside it (fading with a / level, sharpness `halo`) and the contour line."""
    edge = 0.75 * alpha
    inside = edge + (alpha - edge) * np.clip((a - level) / (a.max() - level), 0, 1)
    outside = edge * 0.5 * np.clip(a / level, 0, 1) ** halo
    im = ax.imshow(rgba_field(colour, np.where(a >= level, inside, outside)),
                   extent=(X.min(), X.max(), Y.min(), Y.max()), origin="lower", interpolation="bilinear",
                   zorder=z, aspect="auto")
    cs = ax.contour(X, Y, a, levels=[level], colors=[line_colour], linewidths=line_lw, zorder=z + 0.05)
    if clip is not None:
        im.set_clip_path(clip)
        cs.set_clip_path(clip)


def grid(x0, x1, y0, y1, n=400):
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    return np.meshgrid(xs, ys)


def electrode(ax, tip, side, length, height, face_light, face_dark, edge, lw, tint, tint_colour, tint_r, tint_a, z,
              tint_level, halo, line_colour, line_lw):
    """Triangle pointing at the molecule (tip at `tip`, base `length` away on `side` = -1 left / +1 right),
    filled with a vertical light-to-dark gradient (lit from above) and optionally tinted near the tip."""
    base_x = tip[0] + side * length
    verts = np.array([tip, [base_x, tip[1] + height / 2], [base_x, tip[1] - height / 2]])
    tri = Polygon(verts, closed=True, facecolor="none", edgecolor=edge, lw=lw, zorder=z + 0.2, joinstyle="miter")
    ax.add_patch(tri)
    x0, x1 = sorted((tip[0], base_x))
    y0, y1 = tip[1] - height / 2, tip[1] + height / 2
    X, Y = grid(x0, x1, y0, y1)
    t = (Y - y0) / (y1 - y0)                                              # 0 bottom, 1 top
    img = np.ones(X.shape + (4,))
    img[..., :3] = (np.outer(1 - t.ravel(), to_rgb(face_dark)) + np.outer(t.ravel(), to_rgb(face_light))
                    ).reshape(X.shape + (3,))
    im = ax.imshow(img, extent=(x0, x1, y0, y1), origin="lower", interpolation="bilinear", zorder=z, aspect="auto")
    im.set_clip_path(tri)
    if tint:                                                              # Kondo tint at the tip
        d2 = (X - tip[0]) ** 2 + (Y - tip[1]) ** 2
        shaded(ax, X, Y, np.exp(-d2 / tint_r ** 2), tint_colour, tint_a, tint_level, halo, line_colour, line_lw,
               z + 0.1, clip=tri)
    return verts


def ellipse_ring(c, a_out, b_out, a_in, b_in, n=180):
    """Path of the region between two concentric axis-aligned ellipses (a full ellipse if a_in = b_in = 0)."""
    t = np.linspace(0, 2 * np.pi, n)
    outer = np.column_stack([c[0] + a_out * np.cos(t), c[1] + b_out * np.sin(t)])
    if a_in <= 0:
        return MplPath(outer, closed=True)
    inner = np.column_stack([c[0] + a_in * np.cos(t[::-1]), c[1] + b_in * np.sin(t[::-1])])   # opposite winding
    return MplPath.make_compound_path(MplPath(outer, closed=True), MplPath(inner, closed=True))


def rings(ax, c, rx, ry, n, colour, alpha, line_colour, line_lw, z):
    """n concentric ellipses (circles if rx = ry), all with the same shape, outer radii (rx, ry), equally spaced.
    The bands between them alternate coloured / empty starting from the outermost one (coloured); with an odd n
    the innermost disc is coloured too."""
    for k in range(n, 0, -1):                                   # k-th ellipse from the centre, outermost first
        f_out, f_in = k / n, (k - 1) / n
        if (n - k) % 2 == 0:                                    # band between ellipse k-1 and k: coloured
            ax.add_patch(PathPatch(ellipse_ring(c, rx * f_out, ry * f_out, rx * f_in, ry * f_in),
                                   facecolor=colour, alpha=alpha, edgecolor="none", zorder=z))
        ax.add_patch(Ellipse(c, 2 * rx * f_out, 2 * ry * f_out, facecolor="none", edgecolor=line_colour, lw=line_lw,
                             zorder=z + 0.05))


def circuit(ax, x_base, y, out, y_wire, x_src, x_amp, src_h, src_gap, amp_r, wire_col, wire_lw, font_size,
            src_label, z):
    """Measurement circuit: wires from the outer base of each electrode (at +-x_base, height y) out by `out`, then
    up (y_wire > y) or down (y_wire < y) to y_wire and joined there, with a DC voltage source (long / short plate)
    at x_src and an ammeter (circle with A) at x_amp on that wire. Returns the extreme points."""
    xo = x_base + out
    up = 1 if y_wire > y else -1                                # source label on the outer side of the wire
    kw = dict(color=wire_col, lw=wire_lw, solid_capstyle="round", solid_joinstyle="round", zorder=z)
    ax.plot([-x_base, -xo, -xo, x_src - src_gap / 2], [y, y, y_wire, y_wire], **kw)
    ax.plot([x_src + src_gap / 2, x_amp - amp_r], [y_wire, y_wire], **kw)
    ax.plot([x_amp + amp_r, xo, xo, x_base], [y_wire, y_wire, y, y], **kw)
    # voltage source: long plate (+) on the left, short thick plate (-) on the right
    ax.plot([x_src - src_gap / 2] * 2, [y_wire - src_h / 2, y_wire + src_h / 2], **kw)
    ax.plot([x_src + src_gap / 2] * 2, [y_wire - src_h / 4, y_wire + src_h / 4], **{**kw, "lw": 2 * wire_lw})
    ext = src_h / 2
    if src_label:
        ax.text(x_src, y_wire + up * (src_h / 2 + 0.08), src_label, fontsize=font_size, ha="center",
                va="bottom" if up > 0 else "top", zorder=z)
        ext += 0.28
    # ammeter
    ax.add_patch(Circle((x_amp, y_wire), amp_r, facecolor="white", edgecolor=wire_col, lw=wire_lw, zorder=z))
    ax.text(x_amp, y_wire, "A", fontsize=font_size, ha="center", va="center_baseline", zorder=z + 0.1)
    return [[-xo, y_wire + up * max(ext, amp_r)], [xo, y]]


def offset_label(ax, xy, direction, text, out_pt, size, z):
    """Label `out_pt` points from xy along `direction`, anchored on the side facing xy (never covers it)."""
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) or 1)
    ha = "left" if d[0] > 0.35 else "right" if d[0] < -0.35 else "center"
    va = "bottom" if d[1] > 0.35 else "top" if d[1] < -0.35 else "center"
    return ax.annotate(text, xy, xytext=out_pt * d, textcoords="offset points", fontsize=size, ha=ha, va=va,
                       zorder=z)


def main(out_dir, size_cm, font, transparent, ell_w, ell_h, thick, site_frac, spins, arrow_len, arrow_lw, arrow_head,
         col_up, col_down, mol_face, mol_side, mol_edge, mol_lw, el_length, el_height, el_light, el_dark, el_edge,
         el_lw, el_gap, kondo_site, kondo_lead, kondo_colour, kondo_rx, kondo_ry, kondo_alpha, kondo_rings,
         tint_r, tint_alpha, kondo_level, kondo_halo, kondo_line, kondo_lw, kondo_shift, lead_spins, lead_spin_len,
         lead_spin_lw, lead_spin_head, pad, show_circuit=False, circ=None, label_size=7, j_int=None, j_k=None,
         alt_up=None):
    """alt_up: dict(dots=..., gap=...) draws a dotted red up arrow (solid head) at each down spin, showing that
    it can also point up; None: off."""
    V.set_fonts(font)
    c = np.array([0.0, 0.0])
    dx = site_frac * ell_w / 2
    site = {1: c - [dx, 0], 2: c + [dx, 0]}

    fig, ax = V.new_figure(size_cm)
    slab(ax, c, ell_w, ell_h, thick, mol_face, mol_side, mol_edge, mol_lw, 1)

    pts = []
    tips = {}
    for side, name in ((-1, "left"), (1, "right")):                     # tips touch the ends of the disc
        tip = tips[name] = np.array([side * (ell_w / 2 + el_gap), -thick / 2])
        pts.append(electrode(ax, tip, side, el_length, el_height, el_light, el_dark, el_edge, el_lw,
                             name == kondo_lead, kondo_colour, tint_r, tint_alpha, 0.2, kondo_level, kondo_halo,
                             kondo_line, kondo_lw))

    for i, sgn in spins.items():                                          # spins out of the plane
        col = col_up if sgn > 0 else col_down
        if sgn > 0:
            tip = site[i] + [0, arrow_len]
        else:                                    # down spins come out of the lower side of the disc
            rim = c[1] - thick - ell_h / 2 * np.sqrt(max(0.0, 1 - ((site[i][0] - c[0]) / (ell_w / 2)) ** 2))
            tip = np.array([site[i][0], rim - arrow_len])
        ax.add_patch(FancyArrowPatch(site[i], tip, arrowstyle="-|>", mutation_scale=arrow_head, lw=arrow_lw,
                                     color=col, shrinkA=0, shrinkB=0, zorder=5 if sgn > 0 else 0.5,
                                     capstyle="round"))
        ax.scatter(*site[i], s=14, color=col, edgecolor="white", lw=0.4, zorder=5.5)
        pts.append([tip])
        if alt_up and sgn < 0:                   # alternative orientation: dotted red up arrow, solid head
            up = site[i] + [0, arrow_len]
            head = alt_up["head_len"]
            ax.plot([site[i][0]] * 2, [site[i][1], up[1] - head], color=col_up, lw=arrow_lw,
                    ls=(0, alt_up["dots"]), dash_capstyle="round", zorder=5)
            ax.add_patch(FancyArrowPatch(up - [0, head], up, arrowstyle="-|>", mutation_scale=arrow_head,
                                         lw=arrow_lw, color=col_up, shrinkA=0, shrinkB=0, zorder=5))
            pts.append([up])

    # Kondo area (concentric rings) shared by the screened spin and the tip of the nearest electrode: over the disc
    # and the electrode, under the spin dot and arrows. KONDO_SHIFT moves it from the spin towards the electrode
    # (x mirrored on the right)
    side = -1 if site[kondo_site][0] < c[0] else 1
    centre = site[kondo_site] + np.array([side * -kondo_shift[0], kondo_shift[1]])
    rings(ax, centre, kondo_rx, kondo_ry, kondo_rings, kondo_colour, kondo_alpha, kondo_line, kondo_lw, 1.5)

    # screening electrons: small down spins around that electrode tip (positions relative to the tip, x mirrored)
    lead_tip = tips["left" if side < 0 else "right"]
    for px, py in lead_spins:
        mid = lead_tip + np.array([side * -px, py])
        a, b = mid + [0, lead_spin_len / 2], mid - [0, lead_spin_len / 2]
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=lead_spin_head, lw=lead_spin_lw,
                                     color=col_down, shrinkA=0, shrinkB=0, zorder=4, capstyle="round"))

    labels = []
    if j_int:                                    # intramolecular exchange: curved double arrow between the radicals
        rad = -j_int["rad"] if j_int["side"] == "above" else j_int["rad"]      # arc3: rad < 0 bows up for a -> b
        a, b = site[1], site[2]
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=j_int["style"], mutation_scale=j_int["head"], lw=j_int["lw"],
                                     color="black", connectionstyle=f"arc3,rad={rad}", shrinkA=j_int["shrink"],
                                     shrinkB=j_int["shrink"], zorder=4.5, capstyle="round"))
        m, n = arc_point(a, b, rad, 0.5)
        labels.append(offset_label(ax, m, n, j_int["label"], j_int["label_out"], label_size, 6))
        pts.append([m])
    if j_k:                                      # Kondo coupling: label next to the outer ring
        t = np.radians(j_k["angle"])
        on_ring = centre + [kondo_rx * np.cos(t), kondo_ry * np.sin(t)]
        normal = [np.cos(t) / kondo_rx, np.sin(t) / kondo_ry]
        labels.append(offset_label(ax, on_ring, normal, j_k["label"], j_k["label_out"], label_size, 6))

    if show_circuit:
        pts.append(circuit(ax, ell_w / 2 + el_gap + el_length, -thick / 2, **circ, z=0.1))

    pts = np.vstack(pts + [[c + [sx * ell_w / 2, sy * ell_h / 2 - (thick if sy < 0 else 0)]
                            for sx in (-1, 1) for sy in (-1, 1)]])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    for _ in range(4):                           # grow the limits until the labels (sized in points) fit inside
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect("equal", adjustable="datalim")                      # imshow would reset it
        if not labels:
            break
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        inv = ax.transData.inverted()
        corners = np.vstack([inv.transform([[b.x0, b.y0], [b.x1, b.y1]])
                             for b in (t.get_window_extent(r) for t in labels)])
        new_lo, new_hi = np.minimum(lo, corners.min(axis=0)), np.maximum(hi, corners.max(axis=0))
        if np.allclose(new_lo, lo) and np.allclose(new_hi, hi):
            break
        lo, hi = new_lo, new_hi
    out_dir.mkdir(parents=True, exist_ok=True)
    V.save(fig, out_dir / ("spin_model_junction_circuit" if show_circuit else "spin_model_junction"), transparent)


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    OUT_DIR = HERE / "outputs"
    SIZE_CM = (5.0, 2.5)
    FONT = "Arial"
    TRANSPARENT = True
    PAD = 0.1                                   # margin (drawing units)

    # molecule (same drawing units and style as spin_model_scheme.py)
    ELL_W, ELL_H = 4.0, 1.0
    THICK = 0.12
    SITE_FRAC = 0.72
    SPINS = {1: +1, 2: -1}                      # +1 up (red), -1 down (blue); 1 left, 2 right
    ARROW_LEN = 0.85
    ARROW_LW = 1.1
    ARROW_HEAD = 5
    COL_UP, COL_DOWN = "#C0392B", "#1F4E9A"
    MOL_FACE, MOL_SIDE, MOL_EDGE = "#E8EEF4", "#C5D0DB", "#8C9BAA"
    MOL_LW = 0.4

    # electrodes (gold triangles, tip at the end of the molecule)
    EL_LENGTH = 2.2                             # tip-to-base distance
    EL_HEIGHT = 1.5                             # base height
    EL_LIGHT, EL_DARK = "#F3DE8A", "#D4AF37"    # gradient top / bottom
    EL_EDGE = "#A8862A"
    EL_LW = 0.4
    EL_GAP = 0.0                                # gap between tip and molecule end (0 = touching)

    # Kondo screening
    KONDO_SITE = 1                              # spin screened (1 = left); the rings extend to the nearest tip
    KONDO_SHIFT = (-0.3, 0.0)                   # ring centre relative to the spin (negative x = towards the electrode)
    KONDO_LEAD = None                           # extra tint inside an electrode tip ("left", "right" or None)
    KONDO_COLOUR = "#8E7CC3"                    # soft lavender
    KONDO_RX, KONDO_RY = 1.3, 0.9               # outer ring radii (equal: circles; RY < RX: ellipses in perspective)
    KONDO_RINGS = 5                             # number of circles; bands coloured / empty from the outside in
                                                # (odd: the innermost disc is coloured)
    KONDO_ALPHA = 0.45                          # opacity of the coloured bands
    KONDO_LEVEL = 0.35                          # contour of the electrode tint (fraction of the peak)
    KONDO_HALO = 3                              # sharpness of the soft halo outside the contour (larger = shorter)
    KONDO_LINE = "#5E4A9C"                      # ring / contour line colour
    KONDO_LW = 0.5                              # pt
    TINT_R = 0.7                                # reach of the tint from the electrode tip
    TINT_ALPHA = 0.7
    # screening electrons in the electrode: centres of small down arrows relative to the tip
    # (x < 0 = into the electrode; the triangle half height at depth d is d * EL_HEIGHT / (2 * EL_LENGTH))
    LEAD_SPINS = [(-0.5, 0.0), (-0.9, 0.2), (-0.95, -0.22), (-1.35, 0.0)]
    LEAD_SPIN_LEN = 0.32
    LEAD_SPIN_LW = 0.7
    LEAD_SPIN_HEAD = 3.5

    # measurement circuit (output spin_model_junction_circuit.pdf/.png; SIZE_CM_CIRCUIT used instead of SIZE_CM)
    CIRCUIT = False
    SIZE_CM_CIRCUIT = (6.0, 3.2)
    CIRC = {"out": 0.35,                        # wire length out of each electrode base before turning up / down
            "y_wire": 1.75,                     # height of the source / ammeter wire (> 0: above the junction,
                                                # < 0: below, e.g. -2.0)
            "x_src": -1.3, "x_amp": 1.3,        # positions of the voltage source and the ammeter
            "src_h": 0.55,                      # height of the long plate of the source
            "src_gap": 0.16,                    # distance between the two plates
            "amp_r": 0.28,                      # ammeter radius
            "wire_col": "black", "wire_lw": 0.6,
            "font_size": 7,                     # pt, "A" and the source label
            "src_label": "V"}                   # label on the outer side of the source (None: no label)

    # exchange couplings
    LABEL_SIZE = 7                              # pt
    J_INT = {"side": "above",                   # arc between the two radicals: "above" or "below" the molecule
             "rad": 0.25,                       # curvature (arc3 rad)
             "style": "<|-|>", "lw": 0.55, "head": 3, "shrink": 3,
             "label": r"$J_\mathrm{int}$", "label_out": 1.5}   # label pt from the arc, on its convex side
    J_K = {"angle": 235,                        # deg, position on the outer Kondo ring (0 = right, 90 = top)
           "label": r"$J_\mathrm{K}$", "label_out": 1.5}       # pt from the ring

    # dotted red up arrow at the blue (down) spin: it can point either way
    ALT_UP = {"dots": (0.1, 1.8),               # dotted shaft: (dash, gap) in units of the line width
              "head_len": 0.18}                 # length of the solid arrow head part (drawing units)
    # ALT_UP = None

    main(OUT_DIR, SIZE_CM_CIRCUIT if CIRCUIT else SIZE_CM, FONT, TRANSPARENT, ELL_W, ELL_H, THICK, SITE_FRAC, SPINS, ARROW_LEN, ARROW_LW, ARROW_HEAD,
         COL_UP, COL_DOWN, MOL_FACE, MOL_SIDE, MOL_EDGE, MOL_LW, EL_LENGTH, EL_HEIGHT, EL_LIGHT, EL_DARK, EL_EDGE,
         EL_LW, EL_GAP, KONDO_SITE, KONDO_LEAD, KONDO_COLOUR, KONDO_RX, KONDO_RY, KONDO_ALPHA, KONDO_RINGS,
         TINT_R, TINT_ALPHA, KONDO_LEVEL, KONDO_HALO, KONDO_LINE, KONDO_LW, KONDO_SHIFT, LEAD_SPINS,
         LEAD_SPIN_LEN, LEAD_SPIN_LW, LEAD_SPIN_HEAD, PAD, CIRCUIT, CIRC, LABEL_SIZE, J_INT, J_K, ALT_UP)
