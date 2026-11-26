#!/usr/bin/env python3
"""
Doublet / quartet version of spin_model_stack_junction.py: the same stacked-molecule junction with one spin
removed (REMOVED, default the lower-right site 2). The empty site is a dashed hollow circle and the three
couplings to it are drawn barred (grey dashed arcs without heads, crossed by a short slash, labels greyed).
Output: OUT_DIR/spin_model_stack_junction_doublet.pdf / .png. Settings in the __main__ block.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                          # DFT root: makes visualisation.* importable
from visualisation.paper.figure3.spin_model_stack_junction import main  # noqa: E402


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    OUT_DIR = HERE / "outputs"
    SCALE_CM = 0.58                             # cm per drawing unit (figure-2 junction: ~0.58); sets the size
    WIDTH_CM = 3.5                              # figure width (cm): overrides SCALE_CM (None: use SCALE_CM)
    FONT = "Arial"
    LABEL_SIZE = 7                              # pt, J labels
    TRANSPARENT = True
    PAD = 0.1                                   # margin (drawing units)

    # molecules and spins (as figure1/spin_model_scheme.py)
    ELL_W, ELL_H = 4.0, 1.0
    THICK = 0.12
    SITE_FRAC = 0.72
    SEP = 2.6                                   # vertical distance between the two molecules
    SHIFT_X = 0.6                               # horizontal shift of the upper molecule
    SPINS = {1: +1, 2: -1, 3: +1, 4: -1}        # +1 up (red), -1 down (blue); 1, 2 lower molecule, 3, 4 upper
    ARROW_LEN = 0.85
    ARROW_LW = 1.1
    ARROW_HEAD = 5
    COL_UP, COL_DOWN = "#C0392B", "#1F4E9A"
    MOL_FACE, MOL_SIDE, MOL_EDGE = "#E8EEF4", "#C5D0DB", "#8C9BAA"
    MOL_LW = 0.4

    J_COL = {"intra": "black", "inter": "black"}
    J_LW = 0.55
    J_HEAD = 3
    J_STYLE = "<|-|>"
    J_RAD = {"12": -0.4, "34": -0.4, "13": -0.45, "24": 0.45, "14": -0.15, "23": 0.15}
    LABEL_T = {"12": 0.5, "34": 0.5, "13": 0.5, "24": 0.5, "14": 0.3, "23": 0.3}
    LABEL_OUT = 1.5                             # pt between an arc and its label
    LABEL_NUDGE = {}                            # extra label shifts in pt, e.g. {"13": (-2, 0)}

    # electrodes (as figure2/spin_model_junction.py), touching only molecule CONTACT (1 lower, 2 upper)
    CONTACT = 2
    EL_LENGTH = 1.2                             # short electrodes: more of the 3 cm width for the molecules
    EL_HEIGHT = 1.5
    EL_LIGHT, EL_DARK = "#F3DE8A", "#D4AF37"
    EL_EDGE = "#A8862A"
    EL_LW = 0.4
    EL_GAP = 0.0

    # Kondo screening of the top-left spin (site 3) with the left electrode
    KONDO_SITE = 3
    KONDO_SHIFT = (-0.3, 0.0)                   # ring centre relative to the spin (negative x = towards the electrode)
    KONDO_COLOUR = "#8E7CC3"
    KONDO_RX, KONDO_RY = 1.0, 0.7
    KONDO_RINGS = 5
    KONDO_ALPHA = 0.45
    KONDO_LINE = "#5E4A9C"
    KONDO_LW = 0.5
    LEAD_SPINS = [(-0.35, 0.0), (-0.65, 0.15), (-0.7, -0.17), (-0.98, 0.0)]   # relative to the electrode tip
    LEAD_SPIN_LEN = 0.32
    LEAD_SPIN_LW = 0.7
    LEAD_SPIN_HEAD = 3.5
    J_K = {"angle": 145,                        # deg, position on the outer Kondo ring (0 = right, 90 = top)
           "label": r"$J_\mathrm{K}$", "label_out": 1.5}
    # J_K = None

    # removed spin and barred couplings
    REMOVED = 2                                 # site without spin (1, 2 lower molecule left / right; 3, 4 upper)
    REMOVED_STYLE = {"colour": "#9A9A9A",       # empty site, barred arcs and their labels
                     "lw": 0.55,                # pt, arcs and empty-site circle
                     "ls": (0, (2, 1.5)),       # arc dash pattern
                     "site_r": 0.17,            # radius of the empty-site circle (drawing units)
                     "bar_len": 5.0,            # pt, slash across each barred arc
                     "bar_angle": 60,           # deg from the arc direction
                     "bar_lw": 0.8,
                     "bar_colour": "#9A9A9A",
                     "bar_t": {"12": 0.75, "23": 0.6, "24": 0.25},  # slash position along each arc (0 = lower-numbered site)
                     "labels": True}            # False: no labels on the barred couplings

    main(OUT_DIR, SCALE_CM, FONT, LABEL_SIZE, TRANSPARENT, PAD, ELL_W, ELL_H, THICK, SITE_FRAC, SEP, SHIFT_X, SPINS,
         ARROW_LEN, ARROW_LW, ARROW_HEAD, COL_UP, COL_DOWN, MOL_FACE, MOL_SIDE, MOL_EDGE, MOL_LW, J_COL, J_LW, J_HEAD,
         J_STYLE, J_RAD, LABEL_T, LABEL_OUT, LABEL_NUDGE, CONTACT, EL_LENGTH, EL_HEIGHT, EL_LIGHT, EL_DARK, EL_EDGE,
         EL_LW, EL_GAP, KONDO_SITE, KONDO_COLOUR, KONDO_RX, KONDO_RY, KONDO_ALPHA, KONDO_RINGS, KONDO_LINE, KONDO_LW,
         KONDO_SHIFT, LEAD_SPINS, LEAD_SPIN_LEN, LEAD_SPIN_LW, LEAD_SPIN_HEAD, J_K, WIDTH_CM,
         removed=REMOVED, removed_style=REMOVED_STYLE, out_name="spin_model_stack_junction_doublet")
