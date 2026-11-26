#!/usr/bin/env python3
"""
Schematic of a mechanically controllable break junction (MCBJ), in the flat outlined style of
spin_model_junction.py (meant to sit below it).

Same device as bending_schematic_blender.py: a bent bronze substrate covered by polyimide, two tapered gold
electrodes meeting at a constriction / gap in the middle, two counter-supports on top near the ends and a pushing
rod underneath at the centre (optionally with force arrows), an etched opening in the polyimide under the electrode
tips (CAVITY) and description labels with leader lines (LABELS). The bent strip and the rods are built in 3D and
viewed orthographically from the front (camera in the plane perpendicular to the strip axis x, so the strip is
exactly horizontal and symmetric), looking down by ELEVATION degrees (0 = pure side view); faces are drawn back to
front with thin outlines.

Several options (palette, view, proportions) can be rendered at once: each entry of OPTIONS writes
OUT_DIR/mcbj_schematic_<name>.pdf / .png at exactly SIZE_CM. Settings in the __main__ block.
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import visualisation.paper.figure1.molecule_torsion_view as V  # noqa: E402


def shade(colour, f):
    """Colour darkened (f < 1) or lightened towards white (f > 1)."""
    c = np.array(to_rgb(colour))
    return tuple(c * f) if f <= 1 else tuple(c + (1 - c) * (f - 1))


class Scene:
    def __init__(self, ax, elevation, bend_radius):
        self.ax, self.R = ax, bend_radius
        self.ce, self.se = np.cos(np.radians(elevation)), np.sin(np.radians(elevation))
        self.pts = []

    def bend(self, x, y, z):
        """Wrap the flat strip (x along it, z up) around a cylinder of radius R (axis along y): centre up."""
        th = np.asarray(x) / self.R
        r = self.R + np.asarray(z)
        return np.stack(np.broadcast_arrays(r * np.sin(th), y, r * np.cos(th) - self.R), axis=-1)

    def proj(self, P):
        P = np.atleast_2d(P)
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        uv = np.column_stack([x, z * self.ce + y * self.se])        # front view, tilted down by the elevation
        self.pts.append(uv)
        return uv

    def poly(self, P, face, edge, lw, z):
        self.ax.add_patch(Polygon(self.proj(P), closed=True, facecolor=face, edgecolor=edge, lw=lw, zorder=z,
                                  joinstyle="round"))


def slab(sc, x0, x1, hw, z0, z1, colour, edge, lw, z, light, n=200):
    """Bent slab x0..x1, half width hw(x), thickness z0..z1: faces drawn back to front (top lighter, front base
    colour, ends darker)."""
    xs = np.linspace(x0, x1, n)
    w = np.array([hw(x) for x in xs])
    top_f = sc.bend(xs, -w, z1)
    top_b = sc.bend(xs, w, z1)
    bot_f = sc.bend(xs, -w, z0)
    bot_b = sc.bend(xs, w, z0)
    faces = [
        (np.vstack([bot_f, bot_b[::-1]]), shade(colour, 0.7)),                       # bottom (hidden)
        (np.vstack([bot_b, top_b[::-1]]), shade(colour, 0.7)),                       # back (hidden)
        (np.vstack([bot_f[:1], bot_b[:1], top_b[:1], top_f[:1]]), shade(colour, light["end"])),
        (np.vstack([bot_f[-1:], bot_b[-1:], top_b[-1:], top_f[-1:]]), shade(colour, light["end"])),
        (np.vstack([top_f, top_b[::-1]]), shade(colour, light["top"])),               # top
        (np.vstack([bot_f, top_f[::-1]]), shade(colour, light["front"])),             # front
    ]
    for k, (P, col) in enumerate(faces):
        sc.poly(P, col, edge, lw, z + 0.001 * k)


def rod(sc, x, z_local, r, length, colour, edge, lw, z, light, n=120):
    """Cylinder of radius r with its axis along y (across the strip), centred on the bent surface point."""
    c = np.ravel(sc.bend(x, 0.0, z_local))
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = lambda y: np.column_stack([c[0] + r * np.cos(t), np.full(n, y), c[2] + r * np.sin(t)])
    front, back = ring(-length / 2), ring(length / 2)
    uv = np.vstack([sc.proj(front), sc.proj(back)])
    hull = uv[ConvexHull(uv).vertices]
    sc.ax.add_patch(Polygon(hull, closed=True, facecolor=shade(colour, light["front"]), edgecolor=edge, lw=lw,
                            zorder=z, joinstyle="round"))
    sc.poly(front, shade(colour, light["cap"]), edge, lw, z + 0.01)


def electrode_hw(x, gap, tip_hw, pad_hw, taper):
    d = abs(x) - gap / 2
    return pad_hw if d >= taper else tip_hw + (pad_hw - tip_hw) * d / taper


def opening(sc, rx, ry, depth, t_pi, colour, edge, lw, z, n=160):
    """Etched opening in the polyimide around the gap: elliptical hole (rx along the strip, ry across) in the top
    face, depth `depth`. Drawn as the visible back wall (darker) with the floor (lighter) clipped to the hole."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = rx * np.cos(t), ry * np.sin(t)
    rim = sc.proj(sc.bend(x, y, t_pi))
    hole = Polygon(rim, closed=True, facecolor=shade(colour, 0.5), edgecolor=edge, lw=lw, zorder=z)
    sc.ax.add_patch(hole)
    floor = Polygon(sc.proj(sc.bend(x, y, t_pi - depth)), closed=True, facecolor=shade(colour, 0.72),
                    edgecolor="none", zorder=z + 0.01)
    sc.ax.add_patch(floor)
    floor.set_clip_path(hole)


def circuit(sc, left, right, y_wire, x_src, x_amp, src_h, src_gap, amp_r, ground, ground_x, ground_drop, ground_w,
            wire_col, wire_lw, font_size, src_label, z, ground_side="right"):
    """Measurement circuit in drawing coordinates: posts up from the two electrode contact points `left`, `right`
    (projected) to a wire at height y_wire carrying a DC voltage source (long / short plate) at x_src and an
    ammeter (circle with A) at x_amp; optionally a ground branch from the top of the left or right post
    (ground_side) to ground_x."""
    ax = sc.ax
    kw = dict(color=wire_col, lw=wire_lw, solid_capstyle="round", solid_joinstyle="round", zorder=z)
    xl, xr = left[0], right[0]
    ax.plot([xl, xl, x_src - src_gap / 2], [left[1], y_wire, y_wire], **kw)
    ax.plot([x_src + src_gap / 2, x_amp - amp_r], [y_wire, y_wire], **kw)
    ax.plot([x_amp + amp_r, xr, xr], [y_wire, y_wire, right[1]], **kw)
    for p in (left, right):                                                # contact dots on the pads
        ax.scatter(*p, s=4, color=wire_col, zorder=z + 0.1, lw=0)
    ax.plot([x_src - src_gap / 2] * 2, [y_wire - src_h / 2, y_wire + src_h / 2], **kw)
    ax.plot([x_src + src_gap / 2] * 2, [y_wire - src_h / 4, y_wire + src_h / 4], **{**kw, "lw": 2 * wire_lw})
    pts = [[x_src, y_wire + src_h / 2]]
    if src_label:
        ax.text(x_src, y_wire + src_h / 2 + 0.1, src_label, fontsize=font_size, ha="center", va="bottom", zorder=z)
    ax.add_patch(Circle((x_amp, y_wire), amp_r, facecolor="white", edgecolor=wire_col, lw=wire_lw, zorder=z))
    ax.text(x_amp, y_wire, "A", fontsize=font_size, ha="center", va="center_baseline", zorder=z + 0.1)
    pts += [[x_amp, y_wire + amp_r]]
    if ground:                                                             # ground symbol: three shrinking bars
        yg = y_wire - ground_drop
        xg0 = xr if ground_side == "right" else xl
        ax.plot([xg0, ground_x, ground_x], [y_wire, y_wire, yg], **kw)
        for k, f in enumerate((1.0, 0.62, 0.25)):
            yk = yg - k * 0.13
            ax.plot([ground_x - f * ground_w / 2, ground_x + f * ground_w / 2], [yk, yk], **kw)
        pts += [[ground_x - ground_w / 2, yg - 0.3]]
    sc.pts.append(np.array(pts))


def field_arrow(sc, base, length, angle, width, colour, edge, lw, label, font_size, label_side, z):
    """Magnetic field: broad hollow block arrow from `base` (drawing units) in direction `angle` (deg), labelled."""
    ax = sc.ax
    d = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    n = np.array([-d[1], d[0]])
    hl, hw, sw = 0.38 * length, width, 0.42 * width                     # head length, head / shaft half widths
    P = np.array([base - sw * n, base + d * (length - hl) - sw * n, base + d * (length - hl) - hw * n,
                  base + d * length, base + d * (length - hl) + hw * n, base + d * (length - hl) + sw * n,
                  base + sw * n])
    ax.add_patch(Polygon(P, closed=True, facecolor=colour, edgecolor=edge, lw=lw, zorder=z, joinstyle="miter"))
    sc.pts.append(P)
    if label:
        side = 1 if label_side == "right" else -1
        pos = base + d * length / 2 - side * n * (hw + 0.12)
        ax.text(*pos, label, fontsize=font_size, ha="left" if side > 0 else "right", va="center", zorder=z)


def main(out_file, size_cm, transparent, elevation, bend_radius, L, W, t_sub, t_pi, gap, pad_hw, tip_hw, taper,
         pad_end, rod_r, push_r, rod_x, rod_over, palette, edge_lw, light, forces, force_len, force_lw, force_head,
         pad, font="Arial", font_size=7, cavity=None, labels=None, leader_lw=0.5, field=None, circ=None):
    V.set_fonts(font)
    fig, ax = V.new_figure(size_cm)
    sc = Scene(ax, elevation, bend_radius)
    edge = palette["edge"]

    rod(sc, 0.0, -t_sub - push_r, push_r, W + 2 * rod_over, palette["rod"], edge, edge_lw, 1, light)   # under
    slab(sc, -L / 2, L / 2, lambda x: W / 2, -t_sub, 0.0, palette["substrate"], edge, edge_lw, 2, light)
    slab(sc, -L / 2, L / 2, lambda x: W / 2, 0.0, t_pi, palette["polyimide"], edge, edge_lw, 3, light)
    if cavity:                                                              # etched opening under the tips
        opening(sc, cavity["rx"], cavity["ry"], cavity["depth"], t_pi, palette["polyimide"], edge, edge_lw, 3.5)
    hw = lambda x: electrode_hw(x, gap, tip_hw, pad_hw, taper)
    for s in (-1, 1):                                                       # electrodes: flat on the top face
        xs = np.linspace(gap / 2, pad_end, 200) * s
        w = np.array([hw(x) for x in xs])
        P = np.vstack([sc.bend(xs, -w, t_pi), sc.bend(xs[::-1], w[::-1], t_pi)])
        sc.poly(P, palette["gold"], shade(palette["gold"], 0.6), edge_lw, 4)
    top = t_pi + rod_r
    for s in (-1, 1):
        rod(sc, s * rod_x, top, rod_r, W + 2 * rod_over, palette["rod"], edge, edge_lw, 5 + (s > 0) * 0.1, light)

    if forces:                                                              # push up at the centre, hold the ends
        kw = dict(arrowstyle="-|>", mutation_scale=force_head, lw=force_lw, color=palette["force"], shrinkA=0,
                  shrinkB=0, zorder=6)
        y_front = -(W / 2 + rod_over)
        p = sc.proj(sc.bend(0.0, y_front, -t_sub - 2 * push_r - 0.1))[0]
        ax.add_patch(FancyArrowPatch(p - [0, force_len], p, **kw))
        sc.pts.append(np.array([p - [0, force_len]]))
        for s in (-1, 1):
            q = sc.proj(sc.bend(s * rod_x, W / 2 + rod_over, top + rod_r))[0] + [0, 0.1]   # above the rod top
            ax.add_patch(FancyArrowPatch(q + [0, force_len], q, **kw))
            sc.pts.append(np.array([q + [0, force_len]]))

    # description labels with leader lines (anchor on the part, text offset from it in drawing units)
    anchors = {"polyimide": lambda x: (x, -W / 2, t_pi / 2),
               "electrode": lambda x: (x, 0.0, t_pi),
               "substrate": lambda x: (x, -W / 2, -t_sub / 2),
               "opening": lambda x: (x, 0.0, t_pi),
               "pushing_rod": lambda x: (x, -(W / 2 + rod_over), -t_sub - push_r),
               "counter_support": lambda x: (x, -(W / 2 + rod_over), t_pi + rod_r)}
    if field:                                                               # magnetic field
        base = sc.proj(sc.bend(field["x"], -(W / 2 + rod_over), field["z"]))[0]
        sc.pts.pop()
        field_arrow(sc, base, field["length"], field["angle"], field["width"], field["colour"], field["edge"],
                    edge_lw, field["label"], font_size, field["label_side"], 6)
    if circ:                                                                # measurement circuit
        xp = circ["post_x"]
        cl = sc.proj(sc.bend(-xp, 0.0, t_pi))[0]
        cr = sc.proj(sc.bend(xp, 0.0, t_pi))[0]
        y_top = sc.proj(sc.bend(0.0, W / 2, t_pi))[0][1]
        c = {k: v for k, v in circ.items() if k not in ("post_x", "height")}
        circuit(sc, cl, cr, y_top + circ["height"], **c, font_size=font_size, z=7)

    texts = []
    for text, part, x, (dx, dy), ha in (labels or []):
        a = sc.proj(sc.bend(*anchors[part](x)))[0]
        sc.pts.pop()
        texts.append(ax.annotate(text, a, xytext=a + [dx, dy], textcoords="data", fontsize=font_size, ha=ha,
                                 va="bottom" if dy > 0 else "top", zorder=8,
                                 arrowprops=dict(arrowstyle="-", lw=leader_lw, color="black", shrinkA=1, shrinkB=0)))

    pts = np.vstack(sc.pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    for _ in range(5):                       # grow the limits until the labels (sized in points) fit inside
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        if not texts:
            break
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        inv = ax.transData.inverted()
        corners = np.vstack([inv.transform([[b.x0, b.y0], [b.x1, b.y1]])
                             for b in (t.get_window_extent(r) for t in texts)])
        new_lo, new_hi = np.minimum(lo, corners.min(axis=0)), np.maximum(hi, corners.max(axis=0))
        if np.allclose(new_lo, lo) and np.allclose(new_hi, hi):
            break
        lo, hi = new_lo, new_hi
    V.save(fig, out_file, transparent)


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    OUT_DIR = HERE / "molecule_torsions"
    SIZE_CM = (6.0, 3.0)
    FONT = "Arial"
    FONT_SIZE = 7                                      # pt, labels
    TRANSPARENT = True
    PAD = 0.1

    # geometry (arbitrary units; thicknesses exaggerated for readability)
    GEOM = {"L": 12.0, "W": 2.6,                      # strip length, width
            "t_sub": 0.35, "t_pi": 0.12,              # bronze substrate, polyimide thickness
            "bend_radius": 22.0,                      # radius of curvature (larger = flatter)
            "gap": 0.06, "pad_hw": 0.75, "tip_hw": 0.04, "taper": 2.2, "pad_end": 5.4,   # gold electrodes
            "rod_r": 0.3, "push_r": 0.4,              # counter-supports / pushing rod radius
            "rod_x": 5.5, "rod_over": 0.4}            # counter-support position, rod overhang on each side

    PALETTES = {
        # warm: bronze substrate, amber (Kapton) polyimide, gold electrodes, steel rods
        "warm": {"substrate": "#B9825A", "polyimide": "#E9B15C", "gold": "#E9C85A", "rod": "#A7AEB5",
                 "edge": "#3A3A3A", "force": "black"},
        # neutral: grey steel substrate, pale amber polyimide, gold the only strong colour, dark rods
        "neutral": {"substrate": "#9AA4AE", "polyimide": "#F1D9A6", "gold": "#E3BC3E", "rod": "#5B6168",
                    "edge": "#3A3A3A", "force": "black"},
        # light: almost monochrome light greys, gold electrodes, white rods (lightest, fits a busy figure above)
        "light": {"substrate": "#C9CFD6", "polyimide": "#EEF0F3", "gold": "#E3BC3E", "rod": "#FFFFFF",
                  "edge": "#4A4A4A", "force": "black"},
    }
    LIGHT = {"top": 1.25, "front": 1.0, "end": 0.8, "cap": 1.35}   # face shading (>1 lighter, <1 darker)
    EDGE_LW = 0.4
    FORCE = {"forces": True, "force_len": 0.9, "force_lw": 0.8, "force_head": 5}

    # etched opening in the polyimide under the electrode tips (None: no opening)
    CAVITY = {"rx": 0.9, "ry": 0.65, "depth": 0.1}    # half length (along x), half width (across), depth

    # description labels: (text, part, x position on the part, (dx, dy) text offset from the anchor, alignment)
    # parts: polyimide, electrode, substrate, opening, pushing_rod, counter_support
    LABELS = [("polyimide", "polyimide", -3.0, (0.4, 1.9), "center"),
              ("gold electrodes", "electrode", 2.2, (0.3, 1.3), "center"),
              ("bronze substrate", "substrate", -2.4, (-0.6, -1.3), "center"),
              ("pushing rod", "pushing_rod", 0.35, (1.2, -0.5), "left")]
    LEADER_LW = 0.5

    # magnetic field arrow (None: no field). x, z: base position below the front of the strip (device units);
    # angle in the drawing (90 = up, out of the strip plane); length / width in drawing units
    FIELD = {"x": 5.1, "z": -1.55, "angle": 90, "length": 0.95, "width": 0.24, "colour": "white", "edge": "black",
             "label": r"$\mathbf{B}$", "label_side": "right"}

    # measurement circuit (None: no circuit); SIZE_CM_CIRCUIT is used instead of SIZE_CM
    CIRCUIT = True
    SIZE_CM_CIRCUIT = (6.0, 3.6)
    CIRC = {"post_x": 4.6,                      # wire posts on the electrode pads at +-post_x
            "height": 1.9,                      # wire height above the back edge of the strip (drawing units)
            "x_src": -0.9, "x_amp": 1.3,        # voltage source / ammeter position
            "src_h": 0.5, "src_gap": 0.16, "amp_r": 0.3,
            "ground": True, "ground_side": "right", "ground_x": 5.25, "ground_drop": 0.35, "ground_w": 0.45,
            "wire_col": "black", "wire_lw": 0.6, "src_label": "V"}

    # options to render: name -> (palette, elevation in deg above the strip plane; 0 = pure side view)
    # other options tried: "A_warm": ("warm", 25), "C_light": ("light", 25), "D_neutral_low": ("neutral", 12)
    OPTIONS = {"B_neutral": ("neutral", 25)}

    for name, (pal, el) in OPTIONS.items():
        main(OUT_DIR / f"mcbj_schematic_{name}" if not CIRCUIT else OUT_DIR / f"mcbj_schematic_{name}_circuit",
             SIZE_CM_CIRCUIT if CIRCUIT else SIZE_CM, TRANSPARENT, el, palette=PALETTES[pal],
             edge_lw=EDGE_LW, light=LIGHT, pad=PAD, font=FONT, font_size=FONT_SIZE, cavity=CAVITY, labels=LABELS,
             leader_lw=LEADER_LW, field=FIELD, circ=CIRC if CIRCUIT else None, **GEOM, **FORCE)
