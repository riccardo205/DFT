#!/usr/bin/env python3
"""
Schematic Zeeman diagram of a doublet / quartet pair vs magnetic field (units: g mu_B = 1), in the style of
zeeman_singlet_triplet_quintet.py.

  doublet  E = 0      m = -1/2, +1/2          2 lines
  quartet  E = E_Q    m = -3/2 ... +3/2       4 lines
The ground state d(m=-1/2) meets q(m=-3/2) at B = E_Q. With E_Q > B_MAX the crossing lies outside the plot,
so the doublet stays the ground state over the whole field range; if it falls inside, it is marked with a
dot and a dashed line as in the other diagrams.
Inside the plot, just above the field axis: the ground-state spin label(s). Above the box: regime label(s).
Below the field axis: "Magnetic field". On the energy axis: |d>, |q> at zero field.

The axes box is exactly BOX_CM; the figure is the box plus MARGIN_CM (Arial, transparent).
Output: OUT_DIR/zeeman_doublet_quartet.pdf / .png. Settings in the __main__ block.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                          # DFT root: makes visualisation.* importable
import visualisation.paper.figure1.molecule_torsion_view as V  # noqa: E402
from visualisation.paper.figure1.zeeman_diagram import ket  # noqa: E402


def main(out_dir, box_cm, margin_cm, font, font_size, transparent, e_q, b_max, y_cut, kets, colours, lw,
         mark_colour, mark_lw, mark_ls, mark_alpha, y_margin, y_margin_lo, ylabel_x, ket_style, ket_x, s_pad,
         s_labels, regimes, regime_pad, row_h, xlabel, xlabel_pad):
    V.set_fonts(font)
    l, r, b, t = margin_cm
    W, H = box_cm[0] + l + r, box_cm[1] + b + t
    fig = plt.figure(figsize=(W / 2.54, H / 2.54))
    ax = fig.add_axes((l / W, b / H, box_cm[0] / W, box_cm[1] / H))

    levels = {"d": (0.0, [-0.5, 0.5]), "q": (e_q, [-1.5, -0.5, 0.5, 1.5])}
    B = np.linspace(0, b_max, 400)
    b1 = e_q                                                    # d(-1/2) = q(-3/2)  ->  B = E_Q
    print(f"doublet / quartet crossing at B = {b1:g} ({'inside' if 0 < b1 < b_max else 'outside'} the plot)")
    for k, (e0, ms) in levels.items():
        for m in ms:
            ax.plot(B, e0 + m * B, color=colours[k], lw=lw, solid_capstyle="butt", zorder=2)

    y_lo = min(e0 + min(ms) * b_max for e0, ms in levels.values())
    y_hi = y_cut * e_q if y_cut else max(e0 + max(ms) * b_max for e0, ms in levels.values())
    span = y_hi - y_lo
    ax.set_xlim(0, b_max)
    ax.set_ylim(y_lo - y_margin_lo * span, y_hi + y_margin * span)
    y_axis, y_top = ax.get_ylim()
    data_per_pt = (y_top - y_axis) / (box_cm[1] / 2.54 * 72)

    inside = 0 < b1 < b_max
    xs = {"mid": 0.5 * b_max, "mid0": 0.5 * min(b1, b_max), "mid1": 0.5 * (b1 + b_max), "b1": b1}

    def xpos(x):
        return xs[x] if isinstance(x, str) else x

    if inside:                                                  # crossing: one continuous dashed line
        rows = [g["row"] for g in regimes if g["x"] == "b1"]
        y_end = y_top + ((regime_pad + max(rows) * row_h * font_size - 1.0) * data_per_pt
                         if rows and max(rows) > 0 else 0.0)
        ax.plot([b1, b1], [y_axis, y_end], ls=mark_ls, lw=mark_lw, color=mark_colour, alpha=mark_alpha,
                zorder=1, clip_on=False)
        ax.scatter(b1, -0.5 * b1, s=8, color=mark_colour, zorder=4, lw=0)

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
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=font_size, labelpad=xlabel_pad)

    for k, (e0, _) in levels.items():
        ket(ax, ket_x * b_max, e0, kets[k], colours[k], font_size, **ket_style)

    for g in s_labels:                                          # ground-state spin, just above the field axis
        box = g.get("box")
        ax.annotate(g["text"], (xpos(g["x"]), y_axis), xytext=(g.get("dx", 0), s_pad), textcoords="offset points",
                    fontsize=font_size, ha=g.get("ha", "center"), va="bottom", zorder=5, annotation_clip=False,
                    bbox=None if box is None else dict(boxstyle=f"square,pad={box['pad']}", fc=box["colour"],
                                                       alpha=box["alpha"], ec="none"))
    for g in regimes:                                           # transport regime, above the box
        dy = regime_pad + g["row"] * row_h * font_size
        ax.annotate(g["text"], (xpos(g["x"]), y_top), xytext=(g.get("dx", 0), dy), textcoords="offset points",
                    fontsize=font_size, ha=g.get("ha", "center"), va="bottom", annotation_clip=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    V.save(fig, out_dir / "zeeman_doublet_quartet", transparent)


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    OUT_DIR = HERE / "outputs"
    BOX_CM = (3.33, 2.2)                         # axes box (width, height): 4 cm / 12 T = same field scale as 3 cm / 9 T
    MARGIN_CM = (0.85, 0.3, 0.4, 0.62)          # around the box: left, right, bottom, top (as the s/t/q diagram)
    FONT = "Arial"
    FONT_SIZE = 7                               # pt, all text
    TRANSPARENT = True

    B_CROSS = 9.0                              # field where doublet m=-1/2 and quartet m=-3/2 touch (T)
    E_Q = B_CROSS                               # quartet energy at zero field (crossing at B = E_Q)
    B_MAX = 10.0                                # field range |B| = 0 ... 12 (crossing at 10 visible)
    Y_CUT = 1.15                                # top of the plot at Y_CUT x E_Q (None: show all lines)

    KETS = {"d": "d", "q": "q"}                 # letters in the kets |d>, |q>
    COLOURS = {"d": "#1A1A1A", "q": "#1F4E9A"}  # doublet, quartet
    LW = 1
    MARK_COLOUR = "#1F4E9A"                     # crossing dot / dashed line (only if the crossing is inside)
    MARK_LW = 1.0
    MARK_LS = "--"
    MARK_ALPHA = 1
    Y_MARGIN = 0.03                             # empty space above the lines (fraction of the range)
    Y_MARGIN_LO = 0.2                           # empty space below the lowest line: room for the S labels
    YLABEL_X = 13                               # pt from the energy axis to the right edge of the 'Energy' label
    KET_STYLE = {"lw": 0.55, "height": 1.0, "bracket_w": 0.26, "gap": 0.07, "italic": True}
    KET_X = -0.02                               # right end of the kets (x B_MAX, negative: left of the axis)

    # x: "mid" (centre of the field axis), "mid0" / "mid1" (either side of the crossing), "b1" or a field value
    BOX = {"colour": "white", "alpha": 0.7, "pad": 0.15}
    S_PAD = 2.0                                 # pt between the field axis and the S labels above it
    S_LABELS = [{"text": "S = 1/2", "x": "mid0"},
                {"text": "S = 3/2", "x": "b1", "ha": "left", "dx": 1.5, "box": BOX}]   # range too narrow to centre
    REGIMES = [{"text": "Kondo", "x": 0.0, "row": 1},     # above the box ([] : none): Kondo at 0 T,
               {"text": "IETS", "x": "mid0", "row": 0},   # IETS in between,
               {"text": "Kondo", "x": "b1", "row": 1}]    # Kondo at the crossing (reached by its dashed line)
    REGIME_PAD = 2.0                            # pt between the top of the box and row 0
    ROW_H = 1.15                                # row spacing (x font size)
    XLABEL = "Magnetic field"                   # below the field axis (None: no label)
    XLABEL_PAD = 3                              # pt

    main(OUT_DIR, BOX_CM, MARGIN_CM, FONT, FONT_SIZE, TRANSPARENT, E_Q, B_MAX, Y_CUT, KETS, COLOURS, LW,
         MARK_COLOUR, MARK_LW, MARK_LS, MARK_ALPHA, Y_MARGIN, Y_MARGIN_LO, YLABEL_X, KET_STYLE, KET_X, S_PAD,
         S_LABELS, REGIMES, REGIME_PAD, ROW_H, XLABEL, XLABEL_PAD)
