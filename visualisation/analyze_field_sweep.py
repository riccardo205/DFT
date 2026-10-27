#!/usr/bin/env python3
"""
Electric-field sweep: PBE0/def2-TZVP HS and BS single points on four frozen
geometries, field along the fragment-1 -> fragment-4 axis (e_14) and along the
in-plane perpendicular e_perp (see angle_scan_field_sweep/field_axes.csv).

Sign conventions (same as exchange_analysis.md):
    dE = E_HS - E_BS       dE > 0 : BS lower (AFM tendency)
    F > 0 on "axis"        EField vector points from fragment 1 to fragment 4
    q1 - q4, |S1| - |S4|   fragment-integrated populations

Physical reading of the sign of F is taken from the response itself
(d(q1 - q4)/dF), not assumed.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exchange_convention as XC  # noqa: E402  (J convention, see exchange_convention.py)
from analyze_angle_scan import (  # noqa: E402
    ATOM_LINE_RE,
    FINAL_E_RE,
    HARTREE_TO_MEV,
    MULLIKEN_BLOCK_RE,
    S2_RE,
    build_adjacency,
    outer_terminal_fragment,
    split_fragment_at_bond,
)
from ase.io import read  # noqa: E402

LOEWDIN_RE = re.compile(r"LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS\s*\n-+\n((?:\s*\d+\s+\S+\s*:.*\n)+)")
HIRSH_RE = re.compile(r"HIRSHFELD ANALYSIS.*?ATOM\s+CHARGE\s+SPIN\s*\n((?:\s*\d+\s+\S+\s+-?\d+\.\d+\s+-?\d+\.\d+\s*\n)+)", re.S)
HIRSH_LINE_RE = re.compile(r"\s*(\d+)\s+\S+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
DIPOLE_RE = re.compile(r"Total Dipole Moment\s*:\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
SCF_RE = re.compile(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES")
EFIELD_RE = re.compile(r"EField\s+([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+)")
POINT_RE = re.compile(r"^(axis|perp)_([pm])([\d.]+)$")
STRUCTURES = ["L090_C080_R090", "L135_C120_R090", "L090_C120_R135", "M1"]
SCHEMES = ("mull", "loew", "hirsh")


def _block(text, regex, line_re, charge_col, spin_col):
    m = list(regex.finditer(text))
    if not m:
        return None, None
    q, s = {}, {}
    for line in m[-1].group(1).splitlines():
        lm = line_re.match(line)
        if lm:
            q[int(lm.group(1))] = float(lm.group(charge_col))
            s[int(lm.group(1))] = float(lm.group(spin_col))
    return q, s


def parse_out(path, fragments):
    rec = {"terminated": False}
    if not path.exists():
        return rec
    text = path.read_text(errors="ignore")
    rec["terminated"] = "ORCA TERMINATED NORMALLY" in text
    if not rec["terminated"]:
        return rec
    rec["energy"] = float(FINAL_E_RE.findall(text)[-1])
    rec["s2"] = float(S2_RE.findall(text)[-1])
    scf = SCF_RE.findall(text)
    rec["scf_cycles"] = int(scf[-1]) if scf else np.nan
    rec["trah"] = "TRAH" in text.split("ORCA SCF")[-1] and "Leaving SCF to start the TRAH" in text
    d = DIPOLE_RE.findall(text)
    rec["dipole"] = np.array([float(x) for x in d[-1]]) if d else np.full(3, np.nan)
    blocks = {
        "mull": _block(text, MULLIKEN_BLOCK_RE, ATOM_LINE_RE, 3, 4),
        "loew": _block(text, LOEWDIN_RE, ATOM_LINE_RE, 3, 4),
        "hirsh": _block(text, HIRSH_RE, HIRSH_LINE_RE, 2, 3),
    }
    for scheme, (q, s) in blocks.items():
        for k, atoms in fragments.items():
            rec[f"q{k}_{scheme}"] = sum(q[i] for i in atoms) if q else np.nan
            rec[f"s{k}_{scheme}"] = sum(s[i] for i in atoms) if s else np.nan
    return rec


def load(sweep, fragments):
    axes = pd.read_csv(sweep / "field_axes.csv").set_index("structure")
    rows = []
    for name in STRUCTURES:
        u = {d: axes.loc[name, [f"{d}_x", f"{d}_y", f"{d}_z"]].to_numpy(float) for d in ("axis", "perp")}
        for state, tag in (("high_spin", "HS"), ("broken_symmetry", "BS")):
            sdir = sweep / name / state
            for pdir in sorted(p for p in sdir.iterdir() if p.is_dir()):
                if pdir.name == "F0":
                    dirs, F = ["axis", "perp"], 0.0
                else:
                    m = POINT_RE.match(pdir.name)
                    if not m:
                        continue
                    dirs = [m.group(1)]
                    F = (1 if m.group(2) == "p" else -1) * float(m.group(3))
                    fvec = np.array([float(x) for x in EFIELD_RE.search((pdir / f"{state}.inp").read_text()).groups()])
                    assert np.isclose(fvec @ u[dirs[0]], F, atol=1e-7), f"{pdir}: field not along {dirs[0]}"
                rec = parse_out(pdir / f"{state}.out", fragments)
                for d in dirs:
                    r = {"structure": name, "state": tag, "direction": d, "F_au": F, "point": pdir.name,
                         **{k: v for k, v in rec.items() if k != "dipole"}}
                    if "dipole" in rec:
                        r["mu_axis"] = float(rec["dipole"] @ u["axis"])
                        r["mu_perp"] = float(rec["dipole"] @ u["perp"])
                        r["mu_norm"] = float(np.linalg.norm(rec["dipole"]))
                    rows.append(r)
    return pd.DataFrame(rows)


def pair_table(raw):
    """One row per (structure, direction, F) with HS and BS side by side."""
    ok = raw[raw["terminated"]]
    hs = ok[ok["state"] == "HS"].set_index(["structure", "direction", "F_au"])
    bs = ok[ok["state"] == "BS"].set_index(["structure", "direction", "F_au"])
    idx = hs.index.intersection(bs.index)
    hs, bs = hs.loc[idx], bs.loc[idx]
    t = pd.DataFrame(index=idx)
    t["E_HS"], t["E_BS"] = hs["energy"], bs["energy"]
    t["dE_meV"] = (hs["energy"] - bs["energy"]) * HARTREE_TO_MEV
    t["S2_HS"], t["S2_BS"] = hs["s2"], bs["s2"]
    # H = -J S1.S2 (J > 0 FM), dE = E_HS - E_BS
    t["J_ideal_meV"] = XC.j_ideal(t["dE_meV"])
    t["J_yamaguchi_meV"] = XC.j_yamaguchi(t["dE_meV"], t["S2_HS"], t["S2_BS"])
    t["E_T_minus_E_S_yamaguchi_meV"] = XC.gap_t_minus_s(t["J_yamaguchi_meV"])
    for st, src in (("HS", hs), ("BS", bs)):
        for sc in SCHEMES:
            t[f"dq14_{sc}_{st}"] = src[f"q1_{sc}"] - src[f"q4_{sc}"]
            t[f"dS14_{sc}_{st}"] = src[f"s1_{sc}"].abs() - src[f"s4_{sc}"].abs()
            for k in (1, 2, 3, 4):
                t[f"q{k}_{sc}_{st}"] = src[f"q{k}_{sc}"]
                t[f"s{k}_{sc}_{st}"] = src[f"s{k}_{sc}"]
        t[f"mu_axis_{st}"], t[f"mu_perp_{st}"] = src["mu_axis"], src["mu_perp"]
        t[f"scf_{st}"] = src["scf_cycles"]
    t["bs_signs_ok"] = (np.sign(bs["s1_mull"]) * np.sign(bs["s4_mull"]) < 0) & (bs["s1_mull"].abs() > 0.3) & (bs["s4_mull"].abs() > 0.3)
    return t.reset_index().sort_values(["structure", "direction", "F_au"])


def fragments_from_baseline(repo_root):
    ref = read(repo_root / "scripts" / "orca" / "orca_functional_scan_setup" / "geometries" / "relaxed_main_conf_4.xyz")
    CENTER, LEFT, RIGHT = (4, 9, 17, 18), (42, 41, 0, 5), (21, 23, 28, 29)
    adj = build_adjacency(ref)
    f1 = outer_terminal_fragment(adj, (LEFT[1], LEFT[2]), set(CENTER))
    f4 = outer_terminal_fragment(adj, (RIGHT[1], RIGHT[2]), set(CENTER))
    rest = sorted(set(range(len(ref))) - set(f1) - set(f4))
    f2, f3 = split_fragment_at_bond(adj, (CENTER[1], CENTER[2]), rest)
    return {1: f1, 2: f2, 3: f3, 4: f4}


# ============================================================
# DERIVED QUANTITIES
# ============================================================
AU_TO_V_PER_A = 51.422067          # 1 a.u. of field in V/Angstrom
# Terminal radical whose outer torsion is ~90 deg (decoupled) and the other one.
DECOUPLED = {"L135_C120_R090": (4, 1), "L090_C120_R135": (1, 4)}
STYLE = {  # fixed categorical order (reference palette slots 1-4) + marker as secondary encoding
    "L090_C080_R090": ("#2a78d6", "o", "L090_C080_R090 (sym. 90/90)"),
    "L135_C120_R090": ("#eb6834", "s", "L135_C120_R090 (asym.)"),
    "L090_C120_R135": ("#1baf7a", "D", "L090_C120_R135 (mirror)"),
    "M1": ("#eda100", "^", "M1 (AFM reference)"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def add_relative(t):
    """Changes with respect to F = 0 of the same structure (and decoupled-frame quantities)."""
    t = t.copy()
    zero = t[t["F_au"] == 0].drop_duplicates("structure").set_index("structure")
    cols = ["dE_meV"] + [c for c in t.columns if c.startswith(("dq14_", "dS14_", "q1_", "q2_", "q3_", "q4_", "s1_", "s4_"))]
    for c in cols:
        t[f"d_{c}"] = t[c] - t["structure"].map(zero[c])
    t["F_dc"] = np.nan  # field component pointing from the decoupled to the conjugated radical
    for s, (dc, cj) in DECOUPLED.items():
        m = t["structure"] == s
        sign = 1.0 if dc == 1 else -1.0           # e_14 points 1 -> 4
        t.loc[m & (t["direction"] == "axis"), "F_dc"] = sign * t.loc[m, "F_au"]
        for st in ("HS", "BS"):
            for sc in SCHEMES:
                t.loc[m, f"q_dc_{sc}_{st}"] = t.loc[m, f"q{dc}_{sc}_{st}"]
                t.loc[m, f"q_cj_{sc}_{st}"] = t.loc[m, f"q{cj}_{sc}_{st}"]
                t.loc[m, f"S_dc_{sc}_{st}"] = t.loc[m, f"s{dc}_{sc}_{st}"].abs()
                t.loc[m, f"S_cj_{sc}_{st}"] = t.loc[m, f"s{cj}_{sc}_{st}"].abs()
    return t


PERTURBATIVE_MAX = 0.002   # |F| range used for the linear/quadratic response coefficients


def _odd_even(g, f):
    """Odd and even parts of dE at |F| = f (None if +-f not both present)."""
    yp = g.loc[np.isclose(g["F_au"], f), "dE_meV"]
    ym = g.loc[np.isclose(g["F_au"], -f), "dE_meV"]
    if yp.empty or ym.empty:
        return np.nan, np.nan, np.nan, np.nan
    y0 = g.loc[g["F_au"] == 0, "dE_meV"].iloc[0]
    yp, ym = yp.iloc[0], ym.iloc[0]
    return yp, ym, (yp - ym) / 2, (yp + ym) / 2 - y0


def zero_crossings(g, deg=4):
    """Fields where the full-range polynomial fit of dE(F) crosses zero (within the sampled range)."""
    F, y = g["F_au"].to_numpy(), g["dE_meV"].to_numpy()
    roots = np.roots(np.polyfit(F, y, deg))
    real = [r.real for r in roots if abs(r.imag) < 1e-12 and F.min() <= r.real <= F.max()]
    return sorted(real)


def fit_table(t):
    """Per structure/direction:
    - b, c from dE = a + bF + cF^2 over |F| <= PERTURBATIVE_MAX (linear/quadratic response);
    - odd/even parts at |F| = 0.002 and at the largest sampled |F|;
    - effective power laws odd ~ |F|^n_odd, even ~ |F|^n_even between those two fields;
    - zero crossings of a quartic fit over the full range."""
    rows = []
    for (s, d), g in t.groupby(["structure", "direction"]):
        g = g.sort_values("F_au")
        small = g[g["F_au"].abs() <= PERTURBATIVE_MAX + 1e-12]
        F, y = small["F_au"].to_numpy(), small["dE_meV"].to_numpy()
        c2, b, a = np.polyfit(F, y, 2)
        resid = y - (a + b * F + c2 * F**2)
        fmax = g["F_au"].abs().max()
        y0 = g.loc[g["F_au"] == 0, "dE_meV"].iloc[0]
        _, _, odd2, even2 = _odd_even(g, PERTURBATIVE_MAX)
        yp, ym, oddM, evenM = _odd_even(g, fmax)
        ratio = fmax / PERTURBATIVE_MAX
        pw = lambda hi, lo: np.log(abs(hi / lo)) / np.log(ratio) if abs(lo) > 1e-4 else np.nan  # noqa: E731
        Ff, yf = g["F_au"].to_numpy(), g["dE_meV"].to_numpy()
        resid4 = yf - np.polyval(np.polyfit(Ff, yf, 4), Ff)
        roots = zero_crossings(g)
        rows.append({
            "structure": s, "direction": d, "F_max_au": fmax, "n_points": len(g),
            "dE0_meV": y0, "dE(+Fmax)_meV": yp, "dE(-Fmax)_meV": ym,
            "odd_0.002_meV": odd2, "even_0.002_meV": even2,
            "odd_Fmax_meV": oddM, "even_Fmax_meV": evenM,
            "n_odd": pw(oddM, odd2), "n_even": pw(evenM, even2),
            "rel_change_Fmax_pct": 100 * max(abs(yp - y0), abs(ym - y0)) / abs(y0),
            "slope_meV_per_mau": b * 1e-3, "curv_meV_per_mau2": c2 * 1e-6,
            "fit2_rms_meV(|F|<=0.002)": float(np.sqrt(np.mean(resid**2))),
            "fit4_rms_meV(all)": float(np.sqrt(np.mean(resid4**2))),
            "dE_zero_crossing_au": ";".join(f"{r:+.5f}" for r in roots),
            "dq14_hirsh_per_mau": np.polyfit(F, small["dq14_hirsh_BS"], 1)[0] * 1e-3,
            "alpha_parallel_au": np.polyfit(F, small[f"mu_{d}_BS"], 1)[0],
        })
    return pd.DataFrame(rows)


# ============================================================
# FIGURES
# ============================================================
def _style(ax, xlabel=None, ylabel=None):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK)


def _series(ax, g, x, y, s, label=True, ls="-"):
    col, mk, lab = STYLE[s]
    g = g.sort_values(x)
    ax.plot(g[x], g[y], ls, color=col, lw=2, marker=mk, ms=6, mec="white", mew=1.0, label=lab if label else None)


def _field_axis(ax):
    ax.set_xlim(left=ax.get_xlim()[0], right=ax.get_xlim()[1])
    sec = ax.secondary_xaxis("top", functions=(lambda f: f * AU_TO_V_PER_A, lambda v: v / AU_TO_V_PER_A))
    sec.set_xlabel("F (V/Å)", fontsize=8, color=MUTED)
    sec.tick_params(colors=MUTED, labelsize=7)


def make_figures(t, plots):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots.mkdir(parents=True, exist_ok=True)
    titles = {"axis": "F along e₁₄ (fragment 1 → 4)", "perp": "F along e⊥ (towards backbone)"}

    # --- 1: dE(F) - dE(0) ---
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, d in zip(axs, ("axis", "perp")):
        for s in STRUCTURES:
            _series(ax, t[(t.structure == s) & (t.direction == d)], "F_au", "d_dE_meV", s)
        _style(ax, "F (a.u.)", "ΔE(F) − ΔE(0)  (meV)" if d == "axis" else None)
        ax.set_title(titles[d], fontsize=10, loc="left", color=INK)
        _field_axis(ax)
    axs[0].legend(fontsize=8, frameon=False, loc="upper center")
    fig.text(0.01, 0.005, "ΔE = E_HS − E_BS;  ΔE > 0: BS lower (AFM).  ΔE(0): "
             + ", ".join(f"{s} {t[(t.structure == s) & (t.F_au == 0)].dE_meV.iloc[0]:+.3f}" for s in STRUCTURES) + " meV",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(plots / "fig1_dE_vs_F.png", dpi=200)
    plt.close(fig)

    # --- 2: field-induced charge and spin asymmetry ---
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.6), sharex="col")
    for j, d in enumerate(("axis", "perp")):
        for s in STRUCTURES:
            g = t[(t.structure == s) & (t.direction == d)]
            _series(axs[0, j], g, "F_au", "d_dq14_hirsh_BS", s)
            _series(axs[1, j], g, "F_au", "d_dS14_mull_BS", s)
        _style(axs[0, j], None, "Δ(q₁ − q₄)  (e, Hirshfeld, BS)" if j == 0 else None)
        _style(axs[1, j], "F (a.u.)", "Δ(|S₁| − |S₄|)  (Mulliken, BS)" if j == 0 else None)
        axs[0, j].set_title(titles[d], fontsize=10, loc="left", color=INK)
    axs[0, 0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.text(0.01, 0.005, "Changes relative to F = 0. Absolute q₁ − q₄ is scheme dependent (see text); the field-induced change has the same sign in Mulliken, Löwdin and Hirshfeld.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(plots / "fig2_charge_spin_vs_F.png", dpi=200)
    plt.close(fig)

    # --- 3: mirror pair in the decoupled frame ---
    m = t[t.structure.isin(DECOUPLED) & (t.direction == "axis")]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.3))
    specs = [("d_dE_meV", "ΔE(F) − ΔE(0)  (meV)"),
             ("dq_cj", "Δq (conjugated radical)  (e, Hirshfeld, BS)"),
             ("dS_cj", "Δ|S| (conjugated radical)  (Mulliken, BS)")]
    m = m.copy()
    for s in DECOUPLED:
        k = m.structure == s
        z = m[k & (m.F_au == 0)]
        m.loc[k, "dq_cj"] = m.loc[k, "q_cj_hirsh_BS"] - z["q_cj_hirsh_BS"].iloc[0]
        m.loc[k, "dS_cj"] = m.loc[k, "S_cj_mull_BS"] - z["S_cj_mull_BS"].iloc[0]
        m.loc[k, "dS_dc"] = m.loc[k, "S_dc_mull_BS"] - z["S_dc_mull_BS"].iloc[0]
    for ax, (col, lab) in zip(axs, specs):
        for s in DECOUPLED:
            _series(ax, m[m.structure == s], "F_dc", col, s)
        if col == "dS_cj":
            for s in DECOUPLED:
                _series(ax, m[m.structure == s], "F_dc", "dS_dc", s, label=False, ls=":")
            ax.annotate("dotted: decoupled radical", (0.02, 0.04), xycoords="axes fraction", fontsize=7.5, color=MUTED)
        _style(ax, "F component decoupled → conjugated radical (a.u.)", lab)
    axs[0].legend(fontsize=8, frameon=False, loc="upper right")
    axs[0].set_title("Mirror pair, field along e₁₄ re-expressed relative to the 90° radical", fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(plots / "fig3_mirror_pair_decoupled_frame.png", dpi=200)
    plt.close(fig)

    # --- 4: dE vs field-induced charge / spin asymmetry ---
    a = t[t.direction == "axis"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for s in STRUCTURES:
        g = a[a.structure == s]
        _series(axs[0], g, "d_dq14_hirsh_BS", "d_dE_meV", s)
        _series(axs[1], g, "d_dS14_mull_BS", "d_dE_meV", s)
    _style(axs[0], "Δ(q₁ − q₄)  (e, Hirshfeld, BS)", "ΔE(F) − ΔE(0)  (meV)")
    _style(axs[1], "Δ(|S₁| − |S₄|)  (Mulliken, BS)")
    axs[0].set_title("Field along e₁₄: exchange vs field-induced asymmetry", fontsize=10, loc="left", color=INK)
    axs[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(plots / "fig4_dE_vs_asymmetry.png", dpi=200)
    plt.close(fig)

    # --- 5: QC: linear dipole response and <S^2> ---
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for s in STRUCTURES:
        for d, ls in (("axis", "-"), ("perp", "--")):
            g = t[(t.structure == s) & (t.direction == d)]
            z = g[g.F_au == 0].iloc[0]
            g = g.assign(mu_par=g[f"mu_{d}_BS"] - z[f"mu_{d}_BS"], dS2=1e3 * (g["S2_BS"] - z["S2_BS"]))
            _series(axs[0], g, "F_au", "mu_par", s, label=(d == "axis"), ls=ls)
            _series(axs[1], g, "F_au", "dS2", s, label=False, ls=ls)
            _series(axs[2], g, "F_au", "scf_BS", s, label=False, ls=ls)
    _style(axs[0], "F (a.u.)", "Δμ along F  (a.u.)")
    _style(axs[1], "F (a.u.)", "10³ · Δ⟨S²⟩ BS")
    _style(axs[2], "F (a.u.)", "SCF cycles, BS")
    axs[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    axs[0].set_title("Checks (solid: e₁₄, dashed: e⊥)", fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(plots / "fig5_checks.png", dpi=200)
    plt.close(fig)


def make_highfield_figures(t, fits, plots):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- 6: absolute dE, sign change of the near-degenerate structures ---
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.5))
    near = [s for s in STRUCTURES if s != "M1"]
    a = t[t.direction == "axis"]
    for s in near:
        _series(axs[0], a[a.structure == s], "F_au", "dE_meV", s)
    for s in DECOUPLED:
        g = a[a.structure == s]
        _series(axs[1], g, "F_dc", "dE_meV", s)
        F = np.linspace(g["F_au"].min(), g["F_au"].max(), 400)
        sign = 1.0 if DECOUPLED[s][0] == 1 else -1.0
        axs[1].plot(sign * F, np.polyval(np.polyfit(g["F_au"], g["dE_meV"], 4), F), color=STYLE[s][0], lw=0.8, alpha=0.6)
        for r in zero_crossings(g):
            axs[1].plot(sign * r, 0, "o", ms=9, mfc="none", mec=STYLE[s][0], mew=1.6)
            axs[1].annotate(f"F_dc = {sign * r:+.4f} a.u.\n({sign * r * AU_TO_V_PER_A:+.2f} V/Å)", (sign * r, 0),
                            textcoords="offset points", xytext=(10, -30) if s == "L135_C120_R090" else (14, 14),
                            fontsize=7.5, color=STYLE[s][0])
    for ax in axs:
        ylo, yhi = ax.get_ylim()
        ax.axhspan(0, yhi, color="#f0efec", zorder=-1)
        ax.set_ylim(ylo, yhi)
        ax.text(0.99, 0.97, "BS lower (AFM)", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
        ax.text(0.01, 0.03, "HS lower", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    _style(axs[0], "F along e₁₄ (a.u.)", "ΔE = E_HS − E_BS  (meV)")
    _style(axs[1], "F component decoupled → conjugated radical (a.u.)")
    axs[0].set_title("Near-degenerate structures, field along e₁₄", fontsize=10, loc="left", color=INK)
    axs[1].set_title("Asymmetric pair: sign change of the residual exchange", fontsize=10, loc="left", color=INK)
    axs[0].legend(fontsize=8, frameon=False, loc="upper center")
    fig.text(0.01, 0.005, "Thin lines: quartic fit over all 11 points; open circles: ΔE = 0.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(plots / "fig6_sign_change.png", dpi=200)
    plt.close(fig)

    # --- 7: scaling of the odd and even parts with |F| ---
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for s in STRUCTURES:
        for d, ls in (("axis", "-"), ("perp", "--")):
            g = t[(t.structure == s) & (t.direction == d)]
            fs = sorted(f for f in g["F_au"] if f > 0)
            parts = np.array([_odd_even(g, f)[2:] for f in fs])
            for k, ax in enumerate(axs):
                v = np.abs(parts[:, k])
                keep = v > 2e-4   # below this the part is zero by symmetry / numerical noise
                if keep.sum() >= 2:
                    col, mk, lab = STYLE[s]
                    ax.plot(np.array(fs)[keep], v[keep], ls, color=col, lw=2, marker=mk, ms=6, mec="white", mew=1.0)
    # reference power laws through a point inside the data cloud at |F| = 0.001 a.u.
    xs = np.array([5e-4, 5e-3])
    for ax, (n, y1) in zip(axs, ((1, 2.4e-2), (2, 2.4e-3))):
        ax.set_xscale("log"); ax.set_yscale("log")
        for m in (n, n + 2):
            ys = y1 * (xs / 1e-3) ** m
            ax.plot(xs, ys, ":", color=MUTED, lw=1)
            ax.annotate(f"∝ F$^{m}$", (xs[1], ys[1]), fontsize=8, color=MUTED, textcoords="offset points", xytext=(3, -3))
        ax.set_ylim(1e-4, 1.0)
        ax.grid(True, which="both", color=GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("|F| (a.u.)", fontsize=9, color=INK)
    axs[0].set_ylabel("|part of ΔE(F) − ΔE(0)|  (meV)", fontsize=9, color=INK)
    axs[0].set_title("Odd part [ΔE(F) − ΔE(−F)]/2", fontsize=10, loc="left", color=INK)
    axs[1].set_title("Even part [ΔE(F) + ΔE(−F)]/2 − ΔE(0)", fontsize=10, loc="left", color=INK)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=STYLE[s][0], marker=STYLE[s][1], lw=2, ms=6, mec="white", label=STYLE[s][2]) for s in STRUCTURES]
    handles += [Line2D([], [], color=MUTED, lw=2, ls="-", label="field along e₁₄"),
                Line2D([], [], color=MUTED, lw=2, ls="--", label="field along e⊥")]
    axs[0].legend(handles=handles, fontsize=7.5, frameon=False, loc="upper left")
    fig.text(0.01, 0.005, "Solid: e₁₄, dashed: e⊥. Parts below 2×10⁻⁴ meV (zero by symmetry) omitted.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(plots / "fig7_odd_even_scaling.png", dpi=200)
    plt.close(fig)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    sweep = repo_root / "runs" / "orca" / "short-2OS" / "angle_scans" / "angle_scan_field_sweep"
    fragments = fragments_from_baseline(repo_root)
    raw = load(sweep, fragments)
    raw.to_csv(sweep / "field_sweep_raw.csv", index=False)
    t = add_relative(pair_table(raw))
    t.to_csv(sweep / "field_sweep.csv", index=False)
    fits = fit_table(t)
    fits.to_csv(sweep / "field_sweep_fits.csv", index=False)
    make_figures(t, sweep / "plots")
    make_highfield_figures(t, fits, sweep / "plots")

    n_missing = int((~raw["terminated"]).sum())
    print(f"points parsed: {int(raw['terminated'].sum())} finished, {n_missing} missing/unfinished "
          f"(F0 counted once per direction)")
    print(f"BS terminal spins opposite & |S|>0.3 everywhere: {bool(t['bs_signs_ok'].all())}")
    return raw, t, fits


if __name__ == "__main__":
    raw, t, fits = main()
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(fits.round(4).to_string(index=False))
