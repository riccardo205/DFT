#!/usr/bin/env python3
"""
Schematic Zeeman diagram of a four-spin system: singlet |s>, triplet |t> and quintet |q> levels vs magnetic field.

  E_S,m(B) = E_S + m B        (units: g mu_B = 1)
  singlet  E = 0             m = 0               1 line
  triplet  E = E_T           m = -1, 0, +1       3 lines
  quintet  E = E_Q           m = -2 ... +2       5 lines
Ground-state crossings (marked with a dot and a dashed line down to the field axis):
  singlet / triplet m=-1   at B1 = E_T
  triplet m=-1 / quintet m=-2   at B2 = E_Q - E_T
and the field ranges between them are labelled below the axis with the ground-state spin S = 0, 1, 2.
No ticks or values; the levels are labelled |s>, |t>, |q> at zero field.

Output: OUT_DIR/zeeman_diagram.pdf / .png at exactly SIZE_CM (Arial, transparent). Settings in the __main__ block.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D, ScaledTranslation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import visualisation.paper.figure1.molecule_torsion_view as V  # noqa: E402


def ket(ax, x, y, letter, colour, size, lw=0.55, height=1.0, bracket_w=0.26, gap=0.07, italic=True):
    """|letter> drawn as an Arial letter between a bar and an angle bracket built from lines, so all three
    share the font's height and stroke (mathtext takes | and \\rangle from a fallback font). Right end of
    the ket at (x, y) in data units, vertically centred on y. Sizes in units of the font size (pt)."""
    fig = ax.figure
    pt = fig.dpi / 72.0
    lo, hi = -0.5 * height * size, 0.5 * height * size          # bracket extent around y (pt)
    base = -0.36 * size                                          # letter baseline: x-height centred on y

    def at(dx, dy):                                              # coordinates in points, origin at (x, y) + (dx, dy)
        return (Affine2D().translate(dx, dy).scale(1 / 72) + fig.dpi_scale_trans       # follows the save dpi
                + ScaledTranslation(x, y, ax.transData))

    w = bracket_w * size
    ax.plot([-w, 0, -w], [lo, 0, hi], color=colour, lw=lw, transform=at(0, 0), clip_on=False,
            solid_joinstyle="miter", solid_capstyle="butt")
    xr = -w - gap * size                                         # right edge of the letter
    t = ax.text(0, 0, letter, fontsize=size, color=colour, ha="right", va="baseline",
                style="italic" if italic else "normal", transform=at(xr, base), clip_on=False)
    width = t.get_window_extent(fig.canvas.get_renderer()).width / pt
    xb = xr - width - gap * size                                 # bar
    ax.plot([xb, xb], [lo, hi], color=colour, lw=lw, transform=at(0, 0), clip_on=False, solid_capstyle="butt")


def main(out_dir, size_cm, font, font_size, transparent, e_t, e_q, b_max, colours, lw, gs_lw, highlight_gs,
         mark_colour, ax_box, y_margin, ylabel_pad, y_cut, ket_style):
    V.set_fonts(font)
    levels = {"s": (0.0, [0]), "t": (e_t, [-1, 0, 1]), "q": (e_q, [-2, -1, 0, 1, 2])}
    B = np.linspace(0, b_max, 400)
    b1, b2 = e_t, e_q - e_t                                    # ground-state crossings
    if not 0 < b1 < b2 < b_max:
        print(f"[!] crossings at B = {b1:.2f}, {b2:.2f} not inside (0, {b_max}) in this order")

    fig = plt.figure(figsize=(size_cm[0] / 2.54, size_cm[1] / 2.54))
    ax = fig.add_axes(ax_box)
    for k, (e0, ms) in levels.items():
        for m in ms:
            ax.plot(B, e0 + m * B, color=colours[k], lw=lw, solid_capstyle="butt", zorder=2)
    if highlight_gs:                                            # ground state as a thicker line
        gs = np.min([e0 + m * B for e0, ms in levels.values() for m in ms], axis=0)
        ax.plot(B, gs, color="black", lw=gs_lw, zorder=3, solid_capstyle="butt")

    y_lo = min(e0 + min(ms) * b_max for e0, ms in levels.values())
    y_hi = max(e0 + max(ms) * b_max for e0, ms in levels.values())
    if y_cut:                                                   # cut the top: lines leave through the upper edge
        y_hi = min(y_hi, y_cut * e_q)
    span = y_hi - y_lo
    ax.set_xlim(0, b_max)
    ax.set_ylim(y_lo - y_margin * span, y_hi + y_margin * span)
    y_axis = y_lo - y_margin * span

    # crossings: dot + dashed line down to the field axis
    for b, e in ((b1, 0.0), (b2, e_t - b2)):
        ax.plot([b, b], [y_axis, e], ls=(0, (2, 1.5)), lw=0.5, color=mark_colour, zorder=1)
        ax.scatter(b, e, s=8, color=mark_colour, zorder=4, lw=0)

    # axes: left and bottom with arrow heads, no ticks
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False, ms=2.5)
    ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False, ms=2.5)
    ax.set_xlabel("Magnetic field", fontsize=font_size, labelpad=9)
    ax.set_ylabel("Energy", fontsize=font_size, labelpad=ylabel_pad)

    # level labels at zero field (left of the axis)
    for k, (e0, _) in levels.items():
        ket(ax, -0.02 * b_max, e0, k, colours[k], font_size, **ket_style)
    # ground-state spin between the crossings, below the field axis
    for x0, x1, S in ((0, b1, 0), (b1, b2, 1), (b2, b_max, 2)):
        ax.text(0.5 * (x0 + x1), y_axis, f"S = {S}", fontsize=font_size, ha="center", va="top",
                transform=ax.transData, clip_on=False).set_y(y_axis - 0.02 * span)

    out_dir.mkdir(parents=True, exist_ok=True)
    V.save(fig, out_dir / "zeeman_diagram", transparent)


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    OUT_DIR = HERE / "molecule_torsions"
    SIZE_CM = (4.0, 3.5)
    FONT = "Arial"
    FONT_SIZE = 7                               # pt, all text
    TRANSPARENT = True

    E_T = 0.8                                  # triplet energy at zero field (= E)
    E_Q = 2.4                                   # quintet energy at zero field (= 2.5 E)
    Y_CUT = 1.5                                 # top of the plot at Y_CUT x E_Q (None: show all lines)
    B_MAX = 2.3                                 # field range (crossings at E_T and E_Q - E_T)

    COLOURS = {"s": "#1A1A1A", "t": "#9A9A9A", "q": "#1F4E9A"}   # singlet, triplet, quintet (slate)
    # alternatives: quintet teal "#3E8A86" or ochre "#C08A3E"; triplet taupe "#A08C7A"; previous quintet "#5773A0"
    LW = 0.8
    HIGHLIGHT_GS = False                        # True: ground state drawn on top as a thicker black line
    GS_LW = 1.6
    MARK_COLOUR = "black"                       # crossing dots and dashed lines
    AX_BOX = (0.19, 0.2, 0.78, 0.77)            # axes position in the figure (left, bottom, width, height)
    Y_MARGIN = 0.03                             # empty space above / below the lines (fraction of the range)
    YLABEL_PAD = 14                             # pt between the energy axis and its label (room for the kets)
    KET_STYLE = {"lw": 0.55,                    # pt, stroke of | and the angle bracket
                 "height": 1.0,                 # bracket height (x font size)
                 "bracket_w": 0.26,             # angle bracket width (x font size)
                 "gap": 0.07,                   # space letter - bar / bracket (x font size)
                 "italic": True}

    main(OUT_DIR, SIZE_CM, FONT, FONT_SIZE, TRANSPARENT, E_T, E_Q, B_MAX, COLOURS, LW, GS_LW, HIGHLIGHT_GS,
         MARK_COLOUR, AX_BOX, Y_MARGIN, YLABEL_PAD, Y_CUT, KET_STYLE)
