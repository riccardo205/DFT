#!/usr/bin/env python3
"""
Schematic Zeeman diagram of a singlet / triplet pair vs magnetic field (units: g mu_B = 1).

  singlet  E = 0              m = 0          1 line
  triplet  E = J_int          m = -1, 0, +1  3 lines
The ground state crosses from S = 0 to S = 1 at B = J_int (dot + dashed line down to the field axis).
Inside the plot, just above the field axis: the ground-state spin S = 0 / S = 1 of each field range.
Below the field axis: the transport regime, IETS - Kondo (under the crossing) - IETS. No field label.
On the energy axis: |s>, |t> at zero field and a double arrow J_int between the two levels.
The top of the plot is cut at Y_CUT x E_T.

The axes box is exactly BOX_CM; the figure is the box plus MARGIN_CM (Arial, transparent).
Output: OUT_DIR/zeeman_singlet_triplet.pdf / .png. Settings in the __main__ block.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                          # DFT root: makes visualisation.* importable
import visualisation.paper.figure1.molecule_torsion_view as V  # noqa: E402
from visualisation.paper.figure1.zeeman_diagram import ket  # noqa: E402


def main(out_dir, box_cm, margin_cm, font, font_size, transparent, e_t, b_max, y_cut, colours, lw, mark_colour,
         y_margin, ylabel_x, ket_style, ket_x, s_pad, s_x, s_box, regimes, regime_pad, j_int, mark_lw, mark_ls, mark_alpha):
    V.set_fonts(font)
    l, r, b, t = margin_cm
    W, H = box_cm[0] + l + r, box_cm[1] + b + t
    fig = plt.figure(figsize=(W / 2.54, H / 2.54))
    ax = fig.add_axes((l / W, b / H, box_cm[0] / W, box_cm[1] / H))

    levels = {"s": (0.0, [0]), "t": (e_t, [-1, 0, 1])}
    B = np.linspace(0, b_max, 400)
    b1 = e_t                                                    # singlet / triplet m=-1 crossing
    if not 0 < b1 < b_max:
        print(f"[!] crossing at B = {b1:.2f} not inside (0, {b_max})")
    for k, (e0, ms) in levels.items():
        for m in ms:
            ax.plot(B, e0 + m * B, color=colours[k], lw=lw, solid_capstyle="butt", zorder=2)

    y_lo = e_t - b_max
    y_hi = y_cut * e_t if y_cut else e_t + b_max
    span = y_hi - y_lo
    ax.set_xlim(0, b_max)
    ax.set_ylim(y_lo - y_margin * span, y_hi + y_margin * span)
    y_axis = y_lo - y_margin * span

    # crossing line: ONE line from the dot down to the field axis, or below it to the Kondo label (row > 0),
    # so the dash pattern runs continuously through the axis
    cross_row = regimes["cross"][1] if "cross" in regimes else 0
    y_end = y_axis
    if cross_row > 0:
        drop_pt = regime_pad + cross_row * 1.15 * font_size - 1.0          # pt below the axis (top of the label)
        data_per_pt = (ax.get_ylim()[1] - ax.get_ylim()[0]) / (box_cm[1] / 2.54 * 72)
        y_end = y_axis - drop_pt * data_per_pt
    ax.plot([b1, b1], [0.0, y_end], ls=mark_ls, lw=mark_lw, color=mark_colour, alpha=mark_alpha, zorder=1,
            clip_on=False)
    ax.scatter(b1, 0.0, s=8, color=mark_colour, zorder=4, lw=0)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False, ms=2.5)
    ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False, ms=2.5)
    # energy label placed by hand, just left of the kets (not matplotlib's automatic ylabel position)
    ax.annotate("Energy", (0, 0.5), xycoords="axes fraction", xytext=(-ylabel_x, 0), textcoords="offset points",
                fontsize=font_size, rotation=90, ha="right", va="center", annotation_clip=False)

    for k, (e0, _) in levels.items():
        ket(ax, ket_x * b_max, e0, k, colours[k], font_size, **ket_style)

    # ground-state spin: inside the plot, just above the field axis (x None: centre of its field range)
    for x0, x1, S in ((0, b1, 0), (b1, b_max, 1)):
        x = s_x.get(S) if s_x.get(S) is not None else 0.5 * (x0 + x1)
        box = s_box.get(S)
        ax.annotate(f"S = {S}", (x, y_axis), xytext=(0, s_pad), textcoords="offset points",
                    fontsize=font_size, ha="center", va="bottom", zorder=5,
                    bbox=None if box is None else dict(boxstyle=f"square,pad={box['pad']}", fc=box["colour"],
                                                       alpha=box["alpha"], ec="none"))
    # transport regime: below the field axis, row 0 just under it, row 1 one line lower
    xs = {"left": 0.5 * b1, "cross": b1, "right": 0.5 * (b1 + b_max)}
    row_h = 1.15 * font_size                                    # pt
    for where, (text, row) in regimes.items():
        dy = regime_pad + row * row_h
        ax.annotate(text, (xs[where], y_axis), xytext=(0, -dy), textcoords="offset points",
                    fontsize=font_size, ha="center", va="top", annotation_clip=False)

    # J_int: double arrow left of the energy axis from |s> to |t>
    if j_int:
        x = j_int["x"] * b_max
        ax.annotate("", (x, e_t), (x, 0.0), annotation_clip=False,
                    arrowprops=dict(arrowstyle="<|-|>", lw=j_int["lw"], color=j_int["colour"],
                                    mutation_scale=j_int["head"], shrinkA=0, shrinkB=0))
        ax.annotate(j_int["label"], (x, 0.5 * e_t), xytext=(-j_int["label_pad"], 0), textcoords="offset points",
                    fontsize=font_size, ha="right", va="center", color=j_int["colour"], annotation_clip=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    V.save(fig, out_dir / "zeeman_singlet_triplet", transparent)


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    OUT_DIR = HERE / "outputs"
    BOX_CM = (3.0, 2.2)                         # axes box (width, height)
    MARGIN_CM = (0.85, 0.12, 0.7, 0.12)        # around the box: left, right, bottom, top
    FONT = "Arial"
    FONT_SIZE = 7                               # pt, all text
    TRANSPARENT = True

    E_T = 5.25                                   # triplet energy at zero field (= J_int) = crossing field
    B_MAX = 9.0                                 # field range |B| = 0 ... 9 (crossing at B = E_T)
    Y_CUT = 1.5                                 # top of the plot at Y_CUT x E_T (None: show all lines)

    COLOURS = {"s": "#1A1A1A", "t": "#9A9A9A"}  # singlet, triplet (as in the figure-1 Zeeman diagram)
    LW = 1
    MARK_COLOUR = "#1F4E9A"                       # "#0979AD" crossing dot and dashed line
    MARK_LW = 1.0                               # pt, dashed crossing line
    MARK_LS = "--"                              # as the peak lines of the zero-bias line trace (axvline '--')
    MARK_ALPHA = 1
    Y_MARGIN = 0.03                             # empty space above / below the lines (fraction of the range)
    YLABEL_X = 13                               # pt from the energy axis to the right edge of the 'Energy' label
    KET_STYLE = {"lw": 0.55, "height": 1.0, "bracket_w": 0.26, "gap": 0.07, "italic": True}
    KET_X = -0.02                               # right end of the kets (x B_MAX, negative: left of the axis)
    S_PAD = 2.0                                 # pt between the field axis and the S labels above it
    S_X = {0: None, 1: None}                    # x of the S labels (field units, None: centre of the range = above IETS)
    S_BOX = {1: {"colour": "white", "alpha": 0.7, "pad": 0.15}}   # semi-transparent box behind an S label (None: no box)
    REGIMES = {"left": ("IETS", 0),             # below the field axis: (text, row); row 1 = one line lower
               "cross": ("Kondo", 1),           # under the crossing, reached by the dashed line
               "right": ("IETS", 0)}
    REGIME_PAD = 2.0                            # pt between the field axis and the regime labels
    J_INT = {"x": 0.15,                        # arrow position (x B_MAX, left of the kets)
             "label": r"$J_\mathrm{int}$", "label_pad": 1.5, "lw": 0.5, "head": 4, "colour": "black"}
    # J_INT = None                              # no J_int arrow

    main(OUT_DIR, BOX_CM, MARGIN_CM, FONT, FONT_SIZE, TRANSPARENT, E_T, B_MAX, Y_CUT, COLOURS, LW, MARK_COLOUR,
         Y_MARGIN, YLABEL_X, KET_STYLE, KET_X, S_PAD, S_X, S_BOX, REGIMES, REGIME_PAD, J_INT, MARK_LW, MARK_LS, MARK_ALPHA)
