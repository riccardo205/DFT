#!/usr/bin/env python3
"""
Energy-level diagram of the dimer families: one horizontal line per magnetic family at the lowest energy found
among its dimers, relative to run2/conf_7 (E = 0). The y axis carries only those minimum energies as tick labels.

Data: visualisation/output/orca/dimers_run2/families/dimer_families.csv
  energy  dE_geom_meV (relaxed dimer energy relative to run2/conf_7, the global minimum)
  family  mag_family (classification from the broken-symmetry four-spin fits), mapped onto the figure families
          by FAMILIES; other classes are left out
Only dimers with the relaxation listed in RELAXATIONS are used (default: free relaxations).

The y axis is broken in two (BREAK_LOW: lower window, BREAK_HIGH: upper window) when the families span very
different energies (the closed-shell one is ~1.7 eV up); BREAK = None gives one linear axis.
Output: OUT_DIR/family_energies.pdf / .png at exactly SIZE_CM (Arial, transparent). Settings in __main__.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D, ScaledTranslation
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                          # DFT root: makes visualisation.* importable
import visualisation.paper.figure1.molecule_torsion_view as V  # noqa: E402


def family_minima(csv, families, relaxations, energy_col, e_max=None):
    d = pd.read_csv(csv)
    d = d[d["relaxation"].isin(relaxations)]
    rows = []
    for label, (cls, colour) in families.items():
        sub = d[d["mag_family"] == cls]
        if sub.empty:
            print(f"  [!] no dimers in {cls}")
            continue
        best = sub.loc[sub[energy_col].idxmin()]
        es = sorted(sub[energy_col]) if e_max is None else sorted(e for e in sub[energy_col] if e <= e_max)
        rows.append({"label": label, "colour": colour, "E": best[energy_col], "key": best["key"], "n": len(sub),
                     "all": es})
        print(f"{label!r:<30} {cls:<34} n = {len(sub):2d}   min {best[energy_col]:8.1f} meV  ({best['key']}); "
              f"{len(es)} lines" + ("" if e_max is None else f" <= {e_max:g} meV") + ": "
              + ", ".join(f"{e:.0f}" for e in es))
    return rows


def main(csv, out_dir, size_cm, ax_box, font, font_size, transparent, families, relaxations, energy_col, brk,
         ratio, line_w, line_lw, gap, label_pad, tick_fmt, ylabel, break_mark, ylabel_pad=2.0, ylim=None,
         e_max=None, other_lw=None, label_below=(), legend=None):
    """e_max: draw every dimer of a family up to e_max meV (thin lines, other_lw), the minimum bold (line_lw);
    None: only the minimum."""
    V.set_fonts(font)
    rows = family_minima(csv, families, relaxations, energy_col, e_max)
    n = len(rows)
    fig = plt.figure(figsize=(size_cm[0] / 2.54, size_cm[1] / 2.54))
    l, b, w, h = ax_box
    if brk:
        (lo0, lo1), (hi0, hi1) = brk
        h_lo = (h - gap) * ratio / (1 + ratio)
        axs = [fig.add_axes((l, b + h_lo + gap, w, h - gap - h_lo)), fig.add_axes((l, b, w, h_lo))]
        lims = [(hi0, hi1), (lo0, lo1)]
    else:
        axs, lims = [fig.add_axes((l, b, w, h))], [ylim]

    for k, (ax, lim) in enumerate(zip(axs, lims)):
        for i, r in enumerate(rows):                        # one column per family, line centred in its column
            x0, x1 = i + 0.5 - line_w / 2, i + 0.5 + line_w / 2
            if e_max is not None:            # all dimers of the family up to e_max (thin), minimum bold below
                for e in r["all"]:
                    if e != r["E"]:
                        ax.plot([x0, x1], [e] * 2, color=r["colour"], lw=other_lw, solid_capstyle="butt")
            ax.plot([x0, x1], [r["E"]] * 2, color=r["colour"], lw=line_lw, solid_capstyle="butt")
        ax.set_xlim(0, n)
        if lim:
            ax.set_ylim(*lim)
        ins = [r for r in rows if lim is None or lim[0] <= r["E"] <= lim[1]]
        ax.set_yticks([r["E"] for r in ins])
        ax.set_yticklabels([tick_fmt.format(r["E"]) for r in ins], fontsize=font_size)
        ax.tick_params(axis="y", length=2, width=0.6, pad=1.5)
        ax.set_xticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        if brk:
            ax.spines["bottom" if k == 0 else "top"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.patch.set_alpha(0)
    if legend:                               # families as a list in the top part (line in the family colour)
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=r["colour"], lw=line_lw, solid_capstyle="butt") for r in rows]
        axs[0].legend(handles, [r["label"].replace("\n", "") for r in rows], loc=legend["loc"], ncol=1,
                      frameon=False, fontsize=font_size, handlelength=legend["handlelength"],
                      handletextpad=0.5, labelspacing=legend["labelspacing"], borderaxespad=legend["pad"],
                      bbox_to_anchor=legend.get("anchor"))
    # family labels under the lines (in the lower / only axes) or above them (only without legend)
    for i, r in enumerate([] if legend else rows):
        ax = next(a for a, lim in zip(axs, lims) if lim is None or lim[0] <= r["E"] <= lim[1])
        last = i == n - 1                                  # last label right-aligned to its line end (stays inside)
        below = r["label"] in label_below                  # label under the lowest line instead of above the top
        top = max(r["all"]) if e_max is not None and r["all"] else r["E"]   # label above the highest line
        ax.annotate(r["label"], (i + 0.5 + (line_w / 2 if last else 0), r["E"] if below else top),
                    xytext=(0, -label_pad if below else label_pad),
                    textcoords="offset points", fontsize=font_size, color=r["colour"],
                    ha="right" if last else "center", va="top" if below else "bottom", annotation_clip=False,
                    linespacing=1.0,
                    multialignment="center")
    if brk:                                 # parallel break marks of a fixed size (pt) at the two spine ends
        dx, dy = break_mark
        for ax, ys in ((axs[0], 0), (axs[1], 1)):
            pt = Affine2D().scale(1 / 72) + fig.dpi_scale_trans + ScaledTranslation(0, ys, ax.transAxes)
            ax.plot([-dx, dx], [-dy, dy], transform=pt, color="black", lw=0.6, clip_on=False,
                    solid_capstyle="butt")
    # y label just left of the widest tick label (measured), centred on the whole axis height
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    x_left = min(t.get_window_extent(r).x0 for ax in axs for t in ax.get_yticklabels())
    x_lab = (x_left - ylabel_pad / 72 * fig.dpi) / fig.bbox.width
    lab = fig.text(x_lab, b + h / 2, ylabel, rotation=90, ha="right", va="center", fontsize=font_size)
    if lab.get_window_extent(r).x0 < 0:
        print("  [!] y label cut at the left edge: increase AX_BOX left")
    out_dir.mkdir(parents=True, exist_ok=True)
    V.save(fig, out_dir / "family_energies", transparent)


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    REPO = HERE.parents[2]
    CSV = REPO / "visualisation" / "output" / "orca" / "dimers_run2" / "families" / "dimer_families.csv"
    OUT_DIR = HERE / "outputs"
    SIZE_CM = (3.0, 3.7)
    AX_BOX = (0.34, 0.06, 0.64, 0.86)           # axes area in the figure (left, bottom, width, height)
    FONT = "Arial"
    FONT_SIZE = 7
    TRANSPARENT = True

    # figure family -> (class in mag_family, colour)
    # figure family label -> (class in mag_family, colour); listed in the legend (LEGEND)
    FAMILIES = {"$J_\\mathrm{ext}=J_\\mathrm{FM}$": ("B: dominant FM intermolecular", "#C0392B"),
                "$J_\\mathrm{ext}=J_\\mathrm{AFM}$": ("D: dominant AFM intermolecular", "#1F4E9A"),
                "Network": ("E: competing network", "#5E4A9C")}
    # "Closed-shell": ("F: four-spin mapping fails", "#6E6E6E")   (~1.7 eV; needs BREAK)
    RELAXATIONS = ["free"]                      # add "constrained torsions" to include runs 3 / 4 fixC
    ENERGY_COL = "dE_geom_meV"                  # meV relative to run2/conf_7

    BREAK = None                                # ((lower window), (upper window)) in meV, e.g. ((-20, 260), (1650, 1800))
                                                # with the closed-shell family; None: one linear axis
    YLIM = (-15, 440)                           # meV, range of the linear axis (BREAK = None)
    E_MAX = 300                                 # meV: every dimer of a family up to here as a thin line (None: minima only)
    LEGEND = {"loc": "upper left", "anchor": (0.0, 1.0),   # families listed at the top (None: label on the lines)
              "handlelength": 1.4, "labelspacing": 0.3, "pad": 0.1}
    LABEL_BELOW = ("Network",)                 # families labelled under their lowest line (room in that column)
    OTHER_LW = 0.7                              # pt, those lines (the family minimum keeps LINE_LW)
    RATIO = 3.0                                 # height of the lower window / upper window
    GAP = 0.03                                  # space between the two windows (fraction of the figure)
    BREAK_MARK = (2.5, 1.5)                     # half width, half height of each break mark (pt)
    LINE_W = 0.7                                # line length (fraction of a family column)
    LINE_LW = 1.5                               # pt
    LABEL_PAD = 1.5                             # pt between a line and its family label
    TICK_FMT = "{:.0f}"                         # y tick labels (meV)
    YLABEL = "Energy (meV)"
    YLABEL_PAD = 2.0                            # pt between the tick labels and the y label

    main(CSV, OUT_DIR, SIZE_CM, AX_BOX, FONT, FONT_SIZE, TRANSPARENT, FAMILIES, RELAXATIONS, ENERGY_COL, BREAK,
         RATIO, LINE_W, LINE_LW, GAP, LABEL_PAD, TICK_FMT, YLABEL, BREAK_MARK, YLABEL_PAD, YLIM,
         E_MAX, OTHER_LW, LABEL_BELOW, LEGEND)
