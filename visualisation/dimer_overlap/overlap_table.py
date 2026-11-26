#!/usr/bin/env python3
"""
Geometric overlap vs signed orbital overlap of the dominant intermolecular pair of every dimer.

For every dimer with a dominant pair in the comprehensive analysis (overlap/fragment_pairs.csv,
analyze_dimer_overlap.py) the two magnetic orbitals are computed with magnetic_orbitals.py
(UCO pair of the BS state SPIN_CONFIG, orca_plot cubes cached per dimer; no pictures) and combined with
the geometric descriptors of the same pair. Default state uudd (= dduu, molecule 2 flipped as a whole): every
magnetic UCO pair is intermolecular. In a single-flip state the beta orbital of b pairs with the alpha orbital
it overlaps most, which is the INTRAmolecular partner whenever the intermolecular overlap is small, so the
inter pair is lost there.

  overlap_area_A2   projected fluorenyl ring overlap (geometric, no orbital phases)
  S_grid            |int psi_a psi_b|      net (signed) overlap of the magnetic orbitals = |S|
  coexist           int |psi_a psi_b|      how much the two orbitals occupy the same space (no signs)
  cancellation      1 - S_grid / coexist   0 = all overlap regions have the same sign,
                                           1 = positive and negative regions cancel completely
  S_uco             alpha/beta UCO overlap printed by ORCA for that state (check: = S_grid)
  S2_mag            S^2 of the pair from the 7-state UCO decomposition (comprehensive analysis)
  orb1/2_on_own     fraction of |psi|^2 of each orbital on its own radical half;
  clean             both >= MIN_OWN (otherwise the orbital is shared with another radical and the row is
                    not a clean a-b overlap)
Orbital signs are arbitrary, so only |int psi_a psi_b| is meaningful.

Output: OUT_DIR/overlap_sign_table.csv (one row per dimer). Settings in the __main__ block.
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import magnetic_orbitals as M  # noqa: E402

GEOM_COLS = ["key", "radicals", "J_meV", "reliable", "mag_family", "struct_family", "interplanar_A", "tilt_deg",
             "slip_A", "phi_C2_deg", "d_C9_C9_A", "contact_unit_A", "overlap_area_A2", "overlap_frac",
             "S2_mag", "S_mag"]


def main(pairs_csv, dimers_root, csv_root, out_dir, grid, keys, spin_config, min_own):
    P = pd.read_csv(pairs_csv)
    P = P[P["dominant"] & (P["relation"] == "inter")]
    if keys:
        P = P[P["key"].isin(keys)]
    rows = []
    for _, g in P.iterrows():
        print(f"[{len(rows) + 1}/{len(P)}] {g['key']}")
        try:
            c = M.main(g["key"], dimers_root, csv_root, out_dir, grid, 0.02, 1.0, "x", False, True, 0.09, 1600,
                       False, spin_config, render_images=False)
        except Exception as e:                              # missing state / orca_plot failure: keep the row
            print(f"  [!] {type(e).__name__}: {e}")
            rows.append({**g[GEOM_COLS].to_dict(), "error": str(e)})
            continue
        a, b = g["radicals"]
        s, co = abs(c["int_psi1_psi2"]), c["int_abs_psi1_psi2"]
        on = {a: c["orbital1"], b: c["orbital2"]}
        w = {lab: c[f"{'alpha' if 'alpha' in on[lab] else 'beta'}_on_{lab}"] for lab in (a, b)}
        if c["pair"] != f"J_{g['pair']} ({g['radicals']})":
            print(f"  [!] pair mismatch: {c['pair']} vs {g['pair']}")
        rows.append({**g[GEOM_COLS].to_dict(), "sign_J": "FM" if g["J_meV"] > 0 else "AFM",
                     "state": c["state"], "uco": c["uco"], "S_uco": c["S_uco"],
                     "S_grid": s, "coexist": co, "cancellation": 1 - s / co,
                     "orb1_on_own": w[a], "orb2_on_own": w[b], "clean": min(w.values()) >= min_own, "error": ""})
    T = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    T.to_csv(out_dir / "overlap_sign_table.csv", index=False)
    print(f"wrote {out_dir / 'overlap_sign_table.csv'} ({len(T)} dimers)")


if __name__ == "__main__":
    # =========================
    # USER SETTINGS
    # =========================
    REPO = HERE.parents[1]
    CSV_ROOT = REPO / "visualisation" / "output" / "orca" / "dimers_run2"
    PAIRS_CSV = CSV_ROOT / "overlap" / "fragment_pairs.csv"     # from analyze_dimer_overlap.py
    DIMERS_ROOT = REPO / "runs" / "orca" / "short-2OS" / "dimers_run2"
    OUT_DIR = CSV_ROOT / "dimer_overlap"
    GRID = 100                                                  # orca_plot grid points per axis
    SPIN_CONFIG = "uudd"                                        # BS state (None = only the second radical flipped)
    MIN_OWN = 0.7                                               # localisation threshold of the clean flag
    KEYS = None                                                 # None = every dimer, or e.g. ["run2/conf_7"]

    main(PAIRS_CSV, DIMERS_ROOT, CSV_ROOT, OUT_DIR, GRID, KEYS, SPIN_CONFIG, MIN_OWN)
