#!/usr/bin/env python3
"""
Four-spin model of two stacked molecules (figure1/spin_model_scheme.py) with only three couplings drawn: the two
intramolecular ones J12, J34 and the intermolecular J13 (PAIRS). Same style and drawing units as figure 1.
Output: OUT_DIR/spin_model_scheme_J13.pdf / .png at exactly SIZE_CM. Settings in the __main__ block.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                          # DFT root: makes visualisation.* importable
from visualisation.paper.figure1.spin_model_scheme import main  # noqa: E402


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    OUT_DIR = HERE / "outputs"
    SIZE_CM = (3.5, 3.5)
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
    J_RAD = {"12": 0.2, "34": -0.4, "13": 0.45, "24": 0.45, "14": -0.15, "23": 0.15}
    LABEL_T = {"12": 0.5, "34": 0.5, "13": 0.5, "24": 0.5, "14": 0.3, "23": 0.3}   # label position along each arc
    LABEL_OUT = 1.5                             # pt between the arc and the nearest edge of its label (convex side)
    PAIRS = ("12", "34", "13")                 # couplings drawn: intramolecular J12, J34 + intermolecular J13
    TURN_IN = 15.0                              # deg, both molecules turned further towards the depth axis
                                                # (more compact figure; 0 = the 135 / 45 deg layout)
    AZIMUTH = (135.0 - TURN_IN, 45.0 + TURN_IN)                    # deg, lower / upper molecule turned about the vertical axis
                                                # (anticlockwise seen from above), spins at the molecule ends:
                                                # bottom blue spin to the left, top blue spin to the right
    ALIGN_RED = True                            # compute SHIFT_X so the two red spins (1, 3) line up vertically
    MOL_ASPECT = 0.5                            # in-plane width / length of each molecule (elongated, so the
                                                # turn shows; perspective from ELL_W x ELL_H unturned)
    TEXTS = {"12": r"$J_\mathrm{AFM}$", "34": r"$J_\mathrm{AFM}$",   # labels (default J_ij)
             "13": r"$J_\mathrm{FM}$"}
    LABEL_NUDGE = {"12": (0, 0), "34": (0, 0), "13": (0, 0), "24": (0, 0), "14": (0, 0), "23": (0, 0)}

    if ALIGN_RED:                               # upper molecule shifted so the red sites 1 and 3 share x (centre)
        import numpy as np
        SHIFT_X = SITE_FRAC * ELL_W / 2 * (np.cos(np.radians(AZIMUTH[1])) - np.cos(np.radians(AZIMUTH[0])))
        print(f"SHIFT_X = {SHIFT_X:.3f} (red spins aligned)")

    main(OUT_DIR, SIZE_CM, FONT, LABEL_SIZE, TRANSPARENT, ELL_W, ELL_H, THICK, SITE_FRAC, SEP, SHIFT_X, SPINS,
         ARROW_LEN, ARROW_LW, ARROW_HEAD, COL_UP, COL_DOWN, MOL_FACE, MOL_SIDE, MOL_EDGE, MOL_LW, J_COL, J_LW, J_HEAD,
         J_STYLE, J_RAD, LABEL_T, LABEL_OUT, LABEL_NUDGE, PAD,
         pairs=PAIRS, out_name="spin_model_scheme_J13", texts=TEXTS,
         azimuth=AZIMUTH, mol_aspect=MOL_ASPECT)
