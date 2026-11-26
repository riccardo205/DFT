#!/usr/bin/env python3
"""
Schematic Zeeman diagram of singlet / triplet / quintet levels vs magnetic field (units: g mu_B = 1), in the
style of figure2/zeeman_singlet_triplet.py.

  singlet  E = 0      m = 0           1 line
  triplet  E = E_T    m = -1 ... +1   3 lines
  quintet  E = E_Q    m = -2 ... +2   5 lines
Ground-state crossings (dot + dashed line, continuous through the field axis when a label hangs below it):
  B1 = E_T          singlet / triplet m=-1
  B2 = E_Q - E_T    triplet m=-1 / quintet m=-2
Inside the plot, just above the field axis: the ground-state spin S = 0 / 1 / 2 of each field range.
Below the field axis: transport-regime labels (IETS / Kondo ...), each placed at a range centre or a crossing.
On the energy axis: |s>, |t>, |q> at zero field and a double arrow J_int between |s> and |t>.

The axes box is exactly BOX_CM; the figure is the box plus MARGIN_CM (Arial, transparent).
Output: OUT_DIR/zeeman_singlet_triplet_quintet.pdf / .png. Settings in the __main__ block.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                          # DFT root: makes visualisation.* importable
import visualisation.paper.figure1.molecule_torsion_view as V  # noqa: E402
from visualisation.paper.figure1.zeeman_diagram import ket  # noqa: E402


def main(out_dir, box_cm, margin_cm, font, font_size, transparent, e_t, e_q, b_max, y_cut, colours, lw,
         mark_colour, mark_lw, mark_ls, mark_alpha, y_margin, y_margin_lo, ylabel_x, ket_style, ket_x, s_pad, s_labels,
         regimes, regime_pad, row_h, j_int, xlabel, xlabel_pad):
    V.set_fonts(font)
    l, r, b, t = margin_cm
    W, H = box_cm[0] + l + r, box_cm[1] + b + t
    fig = plt.figure(figsize=(W / 2.54, H / 2.54))
    ax = fig.add_axes((l / W, b / H, box_cm[0] / W, box_cm[1] / H))

    levels = {"s": (0.0, [0]), "t": (e_t, [-1, 0, 1]), "q": (e_q, [-2, -1, 0, 1, 2])}
    B = np.linspace(0, b_max, 400)
    b1, b2 = e_t, e_q - e_t                                     # ground-state crossings
    if not 0 < b1 < b2 < b_max:
        print(f"[!] crossings at B = {b1:.2f}, {b2:.2f} not inside (0, {b_max}) in this order")
    for k, (e0, ms) in levels.items():
        for m in ms:
            ax.plot(B, e0 + m * B, color=colours[k], lw=lw, solid_capstyle="butt", zorder=2)

    y_lo = min(e0 + min(ms) * b_max for e0, ms in levels.values())
    y_hi = y_cut * e_q if y_cut else max(e0 + max(ms) * b_max for e0, ms in levels.values())
    span = y_hi - y_lo
    ax.set_xlim(0, b_max)
    ax.set_ylim(y_lo - y_margin_lo * span, y_hi + y_margin * span)
    y_axis = y_lo - y_margin_lo * span
    data_per_pt = (ax.get_ylim()[1] - ax.get_ylim()[0]) / (box_cm[1] / 2.54 * 72)

    # named x positions: range centres and crossings
    xs = {"mid0": 0.5 * b1, "mid1": 0.5 * (b1 + b2), "mid2": 0.5 * (b2 + b_max), "b1": b1, "b2": b2}

    def xpos(x):
        return xs[x] if isinstance(x, str) else x

    # crossings: ONE dashed line each, from the field axis up through the top of the box, or further up to
    # the regime label above it (row > 0), so the dash pattern is continuous
    y_top = ax.get_ylim()[1]
    for name, e_cross in (("b1", 0.0), ("b2", e_t - b2)):
        rows = [g["row"] for g in regimes if g["x"] == name]
        y_end = y_top
        if rows and max(rows) > 0:
            y_end = y_top + (regime_pad + max(rows) * row_h * font_size - 1.0) * data_per_pt
        ax.plot([xs[name]] * 2, [y_axis, y_end], ls=mark_ls, lw=mark_lw, color=mark_colour, alpha=mark_alpha,
                zorder=1, clip_on=False)
        ax.scatter(xs[name], e_cross, s=8, color=mark_colour, zorder=4, lw=0)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False, ms=2.5, zorder=6)
    ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False, ms=2.5, zorder=6)
    ax.annotate("Energy", (0, 0.5), xycoords="axes fraction", xytext=(-ylabel_x, 0), textcoords="offset points",
                fontsize=font_size, rotation=90, ha="right", va="center", annotation_clip=False)

    for k, (e0, _) in levels.items():
        ket(ax, ket_x * b_max, e0, k, colours[k], font_size, **ket_style)

    # ground-state spin: inside the plot, just above the field axis
    for g in s_labels:
        box = g.get("box")
        ax.annotate(g["text"], (xpos(g["x"]), y_axis), xytext=(g.get("dx", 0), s_pad), textcoords="offset points",
                    fontsize=font_size, ha=g.get("ha", "center"), va="bottom", zorder=5, annotation_clip=False,
                    bbox=None if box is None else dict(boxstyle=f"square,pad={box['pad']}", fc=box["colour"],
                                                       alpha=box["alpha"], ec="none"))
    # transport regime: above the box, row 0 just over its top edge, row n n lines higher
    for g in regimes:
        dy = regime_pad + g["row"] * row_h * font_size
        ax.annotate(g["text"], (xpos(g["x"]), y_top), xytext=(g.get("dx", 0), dy), textcoords="offset points",
                    fontsize=font_size, ha=g.get("ha", "center"), va="bottom", annotation_clip=False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=font_size, labelpad=xlabel_pad)

    # J_int: double arrow from |s> to |t>
    if j_int:
        x = j_int["x"] * b_max
        ax.annotate("", (x, e_t), (x, 0.0), annotation_clip=False,
                    arrowprops=dict(arrowstyle="<|-|>", lw=j_int["lw"], color=j_int["colour"],
                                    mutation_scale=j_int["head"], shrinkA=0, shrinkB=0))
        ax.annotate(j_int["label"], (x, 0.5 * e_t), xytext=(-j_int["label_pad"], 0), textcoords="offset points",
                    fontsize=font_size, ha="right", va="center", color=j_int["colour"], annotation_clip=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    V.save(fig, out_dir / "zeeman_singlet_triplet_quintet", transparent)


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    OUT_DIR = HERE / "outputs"
    BOX_CM = (3.0, 2.2)                         # axes box (width, height)
    MARGIN_CM = (0.85, 0.3, 0.4, 0.62)          # around the box: left, right, bottom, top (regime labels)
    FONT = "Arial"
    FONT_SIZE = 7                               # pt, all text
    TRANSPARENT = True

    B1, B2 = 4.25, 7.75                         # ground-state crossings (T)
    E_T = B1                                    # triplet energy at zero field (= J_int)
    E_Q = B1 + B2                               # quintet energy at zero field (crossing t/q at E_Q - E_T)
    B_MAX = 9.0                                 # field range |B| = 0 ... 9
    Y_CUT = 1.15                                # top of the plot at Y_CUT x E_Q (None: show all lines)

    COLOURS = {"s": "#1A1A1A", "t": "#9A9A9A", "q": "#1F4E9A"}   # as in the figure-1 Zeeman diagram
    LW = 1
    MARK_COLOUR = "#1F4E9A"                     # crossing dots and dashed lines (figure 2: "#1F4E9A")
    MARK_LW = 1.0
    MARK_LS = "--"
    MARK_ALPHA = 1
    Y_MARGIN = 0.03                             # empty space above the lines (fraction of the range)
    Y_MARGIN_LO = 0.2                           # empty space below the lowest line: room for the S labels
    YLABEL_X = 13                               # pt from the energy axis to the right edge of the 'Energy' label
    KET_STYLE = {"lw": 0.55, "height": 1.0, "bracket_w": 0.26, "gap": 0.07, "italic": True}
    KET_X = -0.02                               # right end of the kets (x B_MAX, negative: left of the axis)

    # x: "mid0" / "mid1" / "mid2" (centre of the S = 0 / 1 / 2 range), "b1" / "b2" (crossing) or a field value
    # ha: horizontal alignment at x; dx: extra shift in pt; box: semi-transparent box behind the text
    BOX = {"colour": "white", "alpha": 0.7, "pad": 0.15}
    S_PAD = 2.0                                 # pt between the field axis and the S labels above it
    S_LABELS = [{"text": "S = 0", "x": "mid0"},
                {"text": "S = 1", "x": "mid1", "box": BOX},
                {"text": "S = 2", "x": "b2", "ha": "left", "dx": 1.5, "box": BOX}]   # range too narrow to centre
    REGIMES = [{"text": "IETS", "x": "mid0", "row": 0},
               {"text": "Kondo", "x": "b1", "row": 1},       # above the crossing, reached by its dashed line
               {"text": "IETS", "x": "mid1", "row": 0},
               {"text": "Kondo", "x": "b2", "row": 1},
               {"text": "IETS", "x": "b2", "ha": "left", "dx": 1.5, "row": 0}]
    REGIME_PAD = 2.0                            # pt between the top of the box and row 0 (regimes are above it)
    ROW_H = 1.15                                # row spacing (x font size)
    J_INT = {"x": 0.15,                         # arrow position (x B_MAX)
             "label": r"$J_\mathrm{int}$", "label_pad": 1.5, "lw": 0.5, "head": 4, "colour": "black"}
    J_INT = None                                # no J_int arrow
    XLABEL = "Magnetic field"                   # below the field axis (None: no label)
    XLABEL_PAD = 3                              # pt

    main(OUT_DIR, BOX_CM, MARGIN_CM, FONT, FONT_SIZE, TRANSPARENT, E_T, E_Q, B_MAX, Y_CUT, COLOURS, LW,
         MARK_COLOUR, MARK_LW, MARK_LS, MARK_ALPHA, Y_MARGIN, Y_MARGIN_LO, YLABEL_X, KET_STYLE, KET_X, S_PAD, S_LABELS,
         REGIMES, REGIME_PAD, ROW_H, J_INT, XLABEL, XLABEL_PAD)
