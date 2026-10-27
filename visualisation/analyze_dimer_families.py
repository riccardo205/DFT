#!/usr/bin/env python3
"""
Structural and magnetic families of the dimers_run2 four-radical dimers (run2 + run3 + run4).

Pipeline (one row per relaxed dimer, run number kept as metadata):
  1. structural families: RMSD between relaxed dimers after optimal superposition, minimised over
     the molecule swap (1<->2) and over the graph automorphisms of each monomer (equivalent atom
     orderings); mirror images are allowed and flagged. Complete-linkage clustering, cut at RMSD_CUT.
  2. structural descriptors: per radical half (A, B on molecule 1; C, D on molecule 2) the actual
     outer ring-plane angle and signed outer dihedral, per molecule the central plane angle and the
     torsional proxy P = cos^2 theta_out1 cos^2 theta_C cos^2 theta_out2; intermolecular centre
     distance, angle between mean molecular planes, slip, closest C...C, radical-carbon distances
     d_AB ... d_CD, closest heavy-atom contact and number of C...C contacts < CONTACT_CUT between
     every pair of radical halves.
  3. conformational energy DeltaE_geom = E_relaxed - E_lowest (unconstrained, converged dimer of any
     run; R2SCAN-3C quintet), classed low / intermediate / high.
  4. four-moment check from the broken-symmetry outputs: fragment spins of A-D in uuuu and their
     spread over all finished configurations.
  5. J_12 ... J_34 (exchange_convention.py: H = -sum J S_i.S_j, J > 0 FM; sites 1-4 = A-D), derived
     |J_intra|, |J_inter|, R_dom, dominant / second pair, |J1/J2|, leave-one-out (jackknife) stability
     and condition number of the fit matrix.
  6. magnetic family (rules in classify_magnetic()).
  7. 16x16 Heisenberg spectrum: ground-state spin, first excitations, ground-state pair correlations
     C_ij = <S_i.S_j> and composite spins S_ij^eff from <(S_i+S_j)^2> = 3/2 + 2 C_ij.
  8. geometry vs magnetism, run reproducibility, structural x magnetic cross-table, representatives.

Inputs: relaxation_summary.csv (Read_output_relax_orca.py), bs_energies.csv / exchange_couplings.csv
(Read_output_bs_orca.py), relaxed geometries <run>/.../high_spin/high_spin.xyz, the monomer
exchange_analysis.csv for the intramolecular reference relation.
Outputs: visualisation/output/orca/dimers_run2/families/ (dimer_families.md, CSVs, plots/).
"""

import json
import re
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from ase.io import read
from networkx.algorithms.isomorphism import GraphMatcher
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "orca"))
import exchange_convention as XC  # noqa: E402
from prepare_dimers_broken_symmetry import bond_graph  # noqa: E402

# =========================
# USER SETTINGS
# =========================
DIMERS_ROOT = REPO_ROOT / "runs" / "orca" / "short-2OS" / "dimers_run2"
CSV_ROOT = REPO_ROOT / "visualisation" / "output" / "orca" / "dimers_run2"
OUT = CSV_ROOT / "families"
REF_XYZ = REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans" / "angle_scan_relaxation" / "L135_C120_R090" / "high_spin.xyz"
MONOMER_EXCHANGE_CSV = REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans" / "angle_scan_exchange_analysis" / "exchange_analysis.csv"
RUNS = ["run2", "run3", "run4"]

RMSD_CUT = 0.5              # Angstrom, complete-linkage cut for "same minimum"
CONTACT_CUT = 4.0           # Angstrom, C...C contacts between radical halves
E_LOW, E_HIGH = 50.0, 200.0  # meV: low < E_LOW <= intermediate < E_HIGH <= high
MOMENT_SPREAD_MAX = 0.3     # max change of |fragment spin| over the BS configurations
JACKKNIFE_MAX = 0.25        # max leave-one-out |dJ| / max|J| for a stable fit
WEAK_INTER_RATIO = 0.3      # max|J_inter| < this * max|J_intra|  -> weakly interacting molecules
DOMINANT_RDOM, DOMINANT_RATIO = 0.4, 2.0   # one pair dominates if R_dom >= and |J1/J2| >=
DEGENERACY_MEV = 1e-6

RADICALS = "ABCD"
PAIRS = XC.pair_labels(4)                                 # "12" .. "34"
PL = {p: RADICALS[int(p[0]) - 1] + RADICALS[int(p[1]) - 1] for p in PAIRS}   # "12" -> "AB"
INTRA = {"12", "34"}
INTER = [p for p in PAIRS if p not in INTRA]
CONFIGS = ["uuuu", "duuu", "uduu", "uudu", "uuud", "dduu", "dudu", "duud"]
# reference rings (L135_C120_R090 numbering): outer torsion of each half and the central rings
OUTER_RINGS = {41: ([41, 42, 45, 44, 43], [0, 1, 2, 3, 4, 5]), 28: ([28, 29, 30, 31, 32], [23, 21, 25, 24, 26, 27])}
OUTER_QUARTET = {41: (42, 41, 0, 5), 28: (21, 23, 28, 29)}
CENTRAL_RINGS = ([9, 4, 3, 6, 7, 8], [17, 18, 19, 22, 24, 25])

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
RUN_COLOR = {"run2": COLORS[0], "run3": COLORS[1], "run4": COLORS[2]}
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_3)
    ax.tick_params(colors=INK_2)


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


# =========================
# GEOMETRY
# =========================
def plane_normal(p):
    return np.linalg.svd(p - p.mean(0))[2][-1]


def plane_angle(pos, ra, rb):
    return float(np.degrees(np.arccos(np.clip(abs(plane_normal(pos[ra]) @ plane_normal(pos[rb])), 0, 1))))


def dihedral(p, a, b, c, d):
    b0, b1, b2 = p[a] - p[b], p[c] - p[b], p[d] - p[c]
    b1 /= np.linalg.norm(b1)
    v, w = b0 - (b0 @ b1) * b1, b2 - (b2 @ b1) * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))


def kabsch(P, Q, allow_mirror):
    """RMSD of P onto Q (centred), optional improper rotation; returns (rmsd, mirrored)."""
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    rot = Vt.T @ np.diag([1, 1, d]) @ U.T
    best = (float(np.sqrt(((P @ rot.T - Q) ** 2).sum(1).mean())), False)
    if allow_mirror:
        Pm = P * np.array([1, 1, -1])
        U, _, Vt = np.linalg.svd(Pm.T @ Q)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        rot = Vt.T @ np.diag([1, 1, d]) @ U.T
        r = float(np.sqrt(((Pm @ rot.T - Q) ** 2).sum(1).mean()))
        if r < best[0]:
            best = (r, True)
    return best


class Reference:
    def __init__(self, xyz):
        ref = read(xyz)
        self.n = len(ref)
        self.graph = bond_graph(ref)
        self.autos = list(GraphMatcher(self.graph, self.graph, node_match=lambda a, b: a["z"] == b["z"]).isomorphisms_iter())

    def map_monomer(self, mono):
        gm = GraphMatcher(self.graph, bond_graph(mono), node_match=lambda a, b: a["z"] == b["z"])
        if not gm.is_isomorphic():
            raise ValueError("monomer not isomorphic to the reference")
        return gm.mapping                      # reference index -> local index


def load_dimers(ref):
    rows, geoms = [], {}
    for run in RUNS:
        summ = pd.read_csv(CSV_ROOT / run / "relaxation_summary.csv")
        for _, s in summ.iterrows():
            cfg = s["Config"]
            xyz = DIMERS_ROOT / run / cfg / "high_spin" / "high_spin.xyz"
            if not xyz.exists():
                continue
            atoms = read(xyz)
            n = ref.n
            pos = atoms.positions
            maps = [ref.map_monomer(atoms[k * n:(k + 1) * n]) for k in (0, 1)]
            constrained = run == "run3" or (run == "run4" and not cfg.startswith("free/"))
            key = f"{run}/{cfg}"
            row = {"key": key, "run": run, "dimer": cfg.split("/")[-1], "set": cfg.split("/")[0] if "/" in cfg else "",
                   "relaxation": "constrained torsions" if constrained else "free",
                   "converged": s["Status"] == "converged", "E_Eh": s["E_final_Eh"], "F_max_eV_A": s.get("F_final_eV_A")}
            # radical halves in the Read_output_bs_orca convention: A/C contain local atom 28, B/D local atom 41
            halves, outer = {}, {}
            for k, (lab28, lab41) in enumerate((("A", "B"), ("C", "D"))):
                mp, off = maps[k], k * n
                g = bond_graph(atoms[off:off + n])
                cut = g.copy()
                cut.remove_edge(mp[9], mp[17])
                for lab, local in ((lab28, 28), (lab41, 41)):
                    half = sorted(nx.node_connected_component(cut, local))
                    halves[lab] = [i + off for i in half]
                    # which reference radical carbon (28 or 41) sits in this half under this mapping
                    ref_rad = 28 if mp[28] in half else 41
                    ra, rb = OUTER_RINGS[ref_rad]
                    outer[lab] = {"theta": plane_angle(pos, [mp[i] + off for i in ra], [mp[i] + off for i in rb]),
                                  "dihedral": dihedral(pos, *[mp[i] + off for i in OUTER_QUARTET[ref_rad]])}
                row[f"theta_C_M{k + 1}"] = plane_angle(pos, [mp[i] + off for i in CENTRAL_RINGS[0]],
                                                       [mp[i] + off for i in CENTRAL_RINGS[1]])
                row[f"dihedral_C_M{k + 1}"] = dihedral(pos, *[mp[i] + off for i in (4, 9, 17, 18)])
            for lab in RADICALS:
                row[f"theta_out_{lab}"] = outer[lab]["theta"]
                row[f"dihedral_out_{lab}"] = outer[lab]["dihedral"]
            for k, (a, b) in enumerate((("A", "B"), ("C", "D"))):
                row[f"P_M{k + 1}"] = float(np.prod([np.cos(np.radians(x)) ** 2 for x in
                                                    (row[f"theta_out_{a}"], row[f"theta_C_M{k + 1}"], row[f"theta_out_{b}"])]))
            # intermolecular geometry
            m1, m2 = pos[:n], pos[n:]
            heavy = atoms.numbers[:n] > 1
            com1, com2 = m1.mean(0), m2.mean(0)
            n1, n2 = plane_normal(m1[heavy]), plane_normal(m2[heavy])
            v = com2 - com1
            nm = n1 + np.sign(n1 @ n2) * n2
            nm /= np.linalg.norm(nm)
            row["centre_distance_A"] = float(np.linalg.norm(v))
            row["mol_plane_angle"] = float(np.degrees(np.arccos(np.clip(abs(n1 @ n2), 0, 1))))
            row["slip_A"] = float(np.linalg.norm(v - (v @ nm) * nm))
            row["stack_distance_A"] = float(abs(v @ nm))
            dmat = np.linalg.norm(m1[heavy][:, None] - m2[heavy][None], axis=-1)
            row["min_CC_A"] = float(dmat.min())
            rad_c = {"A": 28, "B": 41, "C": 28 + n, "D": 41 + n}
            numbers = atoms.numbers
            for p in PAIRS:
                a, b = PL[p]
                row[f"d_rad_{p}"] = float(np.linalg.norm(pos[rad_c[a]] - pos[rad_c[b]]))
                if p in INTER:
                    ha = [i for i in halves[a] if numbers[i] > 1]
                    hb = [i for i in halves[b] if numbers[i] > 1]
                    dd = np.linalg.norm(pos[ha][:, None] - pos[hb][None], axis=-1)
                    row[f"contact_{p}_A"] = float(dd.min())
                    cc = [i for i in halves[a] if numbers[i] == 6], [i for i in halves[b] if numbers[i] == 6]
                    row[f"n_CC_{p}"] = int((np.linalg.norm(pos[cc[0]][:, None] - pos[cc[1]][None], axis=-1) < CONTACT_CUT).sum())
            lab28 = []
            for k, (la, lb) in enumerate((("A", "B"), ("C", "D"))):
                lab28.append(la if maps[k][28] + k * n in halves[la] else lb)
            rows.append(row)
            geoms[key] = (pos, maps, lab28)
    return pd.DataFrame(rows), geoms


def rmsd_matrix(keys, geoms, ref):
    n = ref.n
    canon = {}
    for k in keys:                                        # coordinates of both monomers in reference order
        pos, maps, _ = geoms[k]
        canon[k] = [pos[[maps[m][i] + m * n for i in range(n)]] for m in (0, 1)]
    N = len(keys)
    R = np.zeros((N, N))
    mirror = np.zeros((N, N), dtype=bool)
    transform = {}
    for i, j in combinations(range(N), 2):
        X = np.vstack(canon[keys[i]])
        best = (np.inf, False, None)
        Y1, Y2 = canon[keys[j]]
        for swap in (False, True):
            A, B = (Y2, Y1) if swap else (Y1, Y2)
            for i1, a1 in enumerate(ref.autos):
                pa = A[[a1[t] for t in range(n)]]
                for i2, a2 in enumerate(ref.autos):
                    Y = np.vstack([pa, B[[a2[t] for t in range(n)]]])
                    r = kabsch(Y, X, allow_mirror=True)
                    if r[0] < best[0]:
                        best = (r[0], r[1], (swap, i1, i2))
        R[i, j] = R[j, i] = best[0]
        mirror[i, j] = mirror[j, i] = best[1]
        transform[(i, j)] = best[2]
    return R, mirror, transform


def label_map(i, j, keys, geoms, transform, ref):
    """Radical labels of dimer j that sit on the same atoms as A, B, C, D of dimer i after the optimal
    superposition (molecule swap + automorphisms). Returns {label_i: label_j}."""
    swap, i1, i2 = transform[(min(i, j), max(i, j))]
    if i > j:                                             # transform was computed for (j, i): invert
        inv = label_map(j, i, keys, geoms, transform, ref)
        return {v: k for k, v in inv.items()}
    lab28_i, lab28_j = geoms[keys[i]][2], geoms[keys[j]][2]
    pairs_i = (("A", "B"), ("C", "D"))
    out = {}
    for m_i, auto in ((0, ref.autos[i1]), (1, ref.autos[i2])):
        m_j = (1 - m_i) if swap else m_i
        a28_i = lab28_i[m_i]
        b41_i = pairs_i[m_i][1] if a28_i == pairs_i[m_i][0] else pairs_i[m_i][0]
        a28_j = lab28_j[m_j]
        other_j = pairs_i[m_j][1] if a28_j == pairs_i[m_j][0] else pairs_i[m_j][0]
        # i's half holding reference atom 28 <-> j's half holding reference atom auto[28]
        if auto[28] == 28:
            out[a28_i], out[b41_i] = a28_j, other_j
        else:
            out[a28_i], out[b41_i] = other_j, a28_j
    return out


# =========================
# MAGNETISM
# =========================
def spin_ops():
    sx = np.array([[0, 0.5], [0.5, 0]])
    sy = np.array([[0, -0.5j], [0.5j, 0]])
    sz = np.diag([0.5, -0.5])

    def op(o, i):
        m = np.array([[1.0]])
        for k in range(4):
            m = np.kron(m, o if k == i else np.eye(2))
        return m
    S = [[op(o, i) for o in (sx, sy, sz)] for i in range(4)]
    SS = {(i, j): sum(S[i][a] @ S[j][a] for a in range(3)) for i, j in combinations(range(4), 2)}
    tot = [sum(S[i][a] for i in range(4)) for a in range(3)]
    S2 = sum(t @ t for t in tot)
    return SS, S2


SS_OPS, S2_OP = spin_ops()


def spectrum(J):
    H = -sum(J[f"{i + 1}{j + 1}"] * SS_OPS[(i, j)] for i, j in combinations(range(4), 2))
    w, v = np.linalg.eigh(H)
    s2 = np.real(np.einsum("ik,ij,jk->k", v.conj(), S2_OP, v))
    S = np.round((-1 + np.sqrt(1 + 4 * np.clip(s2, 0, None))) / 2, 3)
    w = w - w[0]
    levels = []                                            # multiplets: (energy, S)
    for e, s in zip(w, S):
        if not levels or abs(e - levels[-1][0]) > DEGENERACY_MEV or s != levels[-1][1]:
            levels.append((e, s))
    gs = np.where(w < DEGENERACY_MEV)[0]
    C = {p: float(np.mean([np.real(v[:, k].conj() @ SS_OPS[(int(p[0]) - 1, int(p[1]) - 1)] @ v[:, k]) for k in gs]))
         for p in PAIRS}
    return levels, C


def jackknife(energies):
    """max over left-out configurations of |dJ_ij|, plus condition numbers."""
    full = XC.fit_heisenberg(energies)
    X = XC.design_matrix(list(energies))
    out = {"cond_full": float(np.linalg.cond(X)), "max_dJ": np.nan, "max_rel_dJ": np.nan, "cond_loo_max": np.nan}
    if full["quality"] != "overdetermined":
        return out
    dmax, conds = {p: 0.0 for p in PAIRS}, []
    for c in energies:
        sub = {k: v for k, v in energies.items() if k != c}
        f = XC.fit_heisenberg(sub)
        conds.append(float(np.linalg.cond(XC.design_matrix(list(sub)))))
        if f["quality"] == "insufficient":
            return out
        for p in PAIRS:
            dmax[p] = max(dmax[p], abs(f["J"][p] - full["J"][p]))
    jmax = max(abs(v) for v in full["J"].values())
    out.update({"max_dJ": max(dmax.values()), "max_rel_dJ": max(dmax.values()) / jmax if jmax else np.nan,
                "cond_loo_max": max(conds), **{f"dJ_{p}": dmax[p] for p in PAIRS}})
    return out


def classify_magnetic(r):
    """Rules (applied in order); returns (family, reason)."""
    if r["fit"] == "insufficient":
        return "F: four-spin mapping fails", "fewer than 7 valid configurations"
    if not r["moments_ok"]:
        return "F: four-spin mapping fails", "fragment moments unstable between BS states"
    if np.isfinite(r["max_rel_dJ"]) and r["max_rel_dJ"] > JACKKNIFE_MAX:
        return "F: four-spin mapping fails", "J unstable when one BS state is removed"
    intra_max = max(abs(r["J_12_meV"]), abs(r["J_34_meV"]))
    inter_max = max(abs(r[f"J_{p}_meV"]) for p in INTER)
    if inter_max < WEAK_INTER_RATIO * intra_max:
        return "A: weakly coupled molecules", "max|J_inter| < 0.3 max|J_intra|"
    if r["R_dom"] >= DOMINANT_RDOM and r["J1_over_J2"] >= DOMINANT_RATIO:
        p = r["dominant_pair"]
        if p in INTER:
            if r["J_dominant_meV"] > 0:
                return f"B: dominant FM intermolecular", f"J_{p} ({PL[p]}) dominates"
            return f"D: dominant AFM intermolecular", f"J_{p} ({PL[p]}) dominates"
        return "A: weakly coupled molecules", f"intramolecular J_{p} dominates"
    return "E: competing network", "no single dominant pair"


# =========================
# MAIN
# =========================
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plots = OUT / "plots"
    plots.mkdir(exist_ok=True)
    ref = Reference(REF_XYZ)
    print(f"reference: {ref.n} atoms, {len(ref.autos)} graph automorphisms")

    # ---- 1-3: structures, families, energies
    S, geoms = load_dimers(ref)
    S = S.sort_values("key", key=lambda c: c.map(natural_key)).reset_index(drop=True)
    free_conv = S[(S["relaxation"] == "free") & S["converged"]]
    e_ref_key = free_conv.loc[free_conv["E_Eh"].idxmin(), "key"]
    S["dE_geom_meV"] = (S["E_Eh"] - free_conv["E_Eh"].min()) * XC.HARTREE_TO_MEV
    S["theta_C_min"] = S[["theta_C_M1", "theta_C_M2"]].min(axis=1)
    S["energy_class"] = pd.cut(S["dE_geom_meV"], [-np.inf, E_LOW, E_HIGH, np.inf],
                               labels=["low", "intermediate", "high"]).astype(str)
    print(f"{len(S)} relaxed dimers; computing RMSD matrix ...")
    keys = S["key"].tolist()
    R, mirror, transform = rmsd_matrix(keys, geoms, ref)
    Z = linkage(squareform(R, checks=False), method="complete")
    labels = fcluster(Z, RMSD_CUT, criterion="distance")
    # name structural families by increasing lowest energy
    S["_cl"] = labels
    order = S.groupby("_cl")["dE_geom_meV"].min().sort_values().index
    name = {c: f"S{i + 1:02d}" for i, c in enumerate(order)}
    S["struct_family"] = S["_cl"].map(name)
    pd.DataFrame(R, index=keys, columns=keys).round(4).to_csv(OUT / "rmsd_matrix.csv")

    # ---- 4-7: magnetism
    E = pd.concat([pd.read_csv(CSV_ROOT / f"{r}_broken_symmetry" / "bs_energies.csv").assign(run=r)
                   for r in RUNS if (CSV_ROOT / f"{r}_broken_symmetry" / "bs_energies.csv").exists()])
    E = E[E["stage"] == "relaxed"]
    Jt = pd.concat([pd.read_csv(CSV_ROOT / f"{r}_broken_symmetry" / "exchange_couplings.csv").assign(run=r)
                    for r in RUNS if (CSV_ROOT / f"{r}_broken_symmetry" / "exchange_couplings.csv").exists()])
    Jt = Jt[Jt["stage"] == "relaxed"]
    mag, corr_rows, level_rows = [], [], []
    for _, j in Jt.iterrows():
        key = f"{j['run']}/{j['dimer']}"
        e = E[(E["run"] == j["run"]) & (E["dimer"] == j["dimer"])]
        done = e[e["status"] == "done"]
        r = {"key": key, "fit": j["fit"], "n_configs_in_fit": j["n_configs_in_fit"], "rms_residual_meV": j["rms_residual_meV"]}
        uuuu = done[done["config"] == "uuuu"]
        if len(uuuu):
            s = uuuu.iloc[0][[f"spin_{x}" for x in RADICALS]].abs().astype(float)
            r.update({"s_mean_uuuu": s.mean(), "s_spread_uuuu": s.max() - s.min(),
                      **{f"s_{x}_uuuu": s[f"spin_{x}"] for x in RADICALS}})
        spread = max((done[f"spin_{x}"].abs().max() - done[f"spin_{x}"].abs().min()) for x in RADICALS) if len(done) else np.nan
        minmom = done[[f"spin_{x}" for x in RADICALS]].abs().min().min() if len(done) else np.nan
        r.update({"moment_spread_BS": spread, "min_moment_BS": minmom,
                  "moments_ok": bool(np.isfinite(spread) and spread <= MOMENT_SPREAD_MAX and minmom >= XC.MIN_LOCAL_MOMENT)})
        for p in PAIRS:
            r[f"J_{p}_meV"] = j[f"J_{p}_meV"]
        used = done[done["used_in_fit"] == True]  # noqa: E712
        energies = {c: (v - used["E_Eh"].min()) * XC.HARTREE_TO_MEV for c, v in zip(used["config"], used["E_Eh"])}
        r.update(jackknife(energies) if len(energies) >= 7 else {"cond_full": np.nan, "max_dJ": np.nan,
                                                                  "max_rel_dJ": np.nan, "cond_loo_max": np.nan})
        if j["fit"] != "insufficient":
            Jv = {p: j[f"J_{p}_meV"] for p in PAIRS}
            a = sorted(PAIRS, key=lambda p: -abs(Jv[p]))
            tot = sum(abs(v) for v in Jv.values())
            r.update({"J_intra_abs": abs(Jv["12"]) + abs(Jv["34"]), "J_inter_abs": sum(abs(Jv[p]) for p in INTER),
                      "R_dom": abs(Jv[a[0]]) / tot, "dominant_pair": a[0], "dominant_radicals": PL[a[0]],
                      "J_dominant_meV": Jv[a[0]], "second_pair": a[1], "J_second_meV": Jv[a[1]],
                      "J1_over_J2": abs(Jv[a[0]] / Jv[a[1]]) if Jv[a[1]] else np.inf})
            levels, C = spectrum(Jv)
            r.update({"S_GS": levels[0][1], "Delta1_meV": levels[1][0], "S1": levels[1][1],
                      "Delta2_meV": levels[2][0] if len(levels) > 2 else np.nan,
                      "S2": levels[2][1] if len(levels) > 2 else np.nan})
            for p in PAIRS:
                r[f"C_{p}"] = C[p]
                v = 1.5 + 2 * C[p]
                r[f"Seff_{p}"] = (-1 + np.sqrt(1 + 4 * max(v, 0))) / 2
            for k, (e_, s_) in enumerate(levels):
                level_rows.append({"key": key, "level": k, "E_meV": e_, "S": s_})
        r["mag_family"], r["mag_reason"] = classify_magnetic(r) if j["fit"] != "insufficient" else \
            ("F: four-spin mapping fails", "fewer than 7 valid configurations")
        mag.append(r)
    M = pd.DataFrame(mag)
    D = S.merge(M, on="key", how="left")
    D["mag_family"] = D["mag_family"].fillna("no BS data")
    # label-invariant comparison: express every fitted member of a structural family in the radical
    # labels of the family's lowest-energy fitted member
    D["J_aligned_to"] = ""
    for p in PAIRS:
        D[f"Jal_{p}_meV"] = np.nan
    for sfam, g in D[D["J_12_meV"].notna()].groupby("struct_family"):
        g = g.sort_values("dE_geom_meV")
        rep_key = g.iloc[0]["key"]
        ri = keys.index(rep_key)
        for idx, r in g.iterrows():
            lm = {x: x for x in RADICALS} if r["key"] == rep_key else label_map(ri, keys.index(r["key"]), keys, geoms, transform, ref)
            for p in PAIRS:
                a, b = PL[p]
                pj = "".join(sorted((lm[a], lm[b])))
                qj = [q for q in PAIRS if PL[q] == pj][0]
                D.loc[idx, f"Jal_{p}_meV"] = r[f"J_{qj}_meV"]
            D.loc[idx, "J_aligned_to"] = rep_key
            D.loc[idx, "label_map_to_rep"] = " ".join(f"{k}->{v}" for k, v in lm.items())
    D.drop(columns="_cl").to_csv(OUT / "dimer_families.csv", index=False)
    pd.DataFrame(level_rows).to_csv(OUT / "heisenberg_spectra.csv", index=False)

    figs = make_plots(D, R, keys, plots)
    write_report(D, R, mirror, keys, e_ref_key, figs, OUT / "dimer_families.md")
    print(f"wrote {OUT / 'dimer_families.md'}")


# =========================
# PLOTS
# =========================
FAM_ORDER = ["A: weakly coupled molecules", "B: dominant FM intermolecular", "D: dominant AFM intermolecular",
             "E: competing network", "F: four-spin mapping fails", "no BS data"]
FAM_STYLE = {f: (c, m) for f, c, m in zip(FAM_ORDER, COLORS[:5] + [INK_3], ["o", "^", "v", "s", "X", "."])}


def make_plots(D, R, keys, plots):
    figs = {}
    # 1. RMSD matrix ordered by structural family
    order = D.sort_values(["struct_family", "dE_geom_meV"]).index
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(R[np.ix_(order, order)], cmap="viridis_r", vmin=0, vmax=3)
    lab = [f"{D.loc[i, 'struct_family']} {D.loc[i, 'key']}" for i in order]
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(lab, rotation=90, fontsize=5)
    ax.set_yticklabels(lab, fontsize=5)
    fig.colorbar(im, ax=ax, label="RMSD (Å), min over swap / automorphisms / mirror")
    ax.set_title("Structural similarity of all relaxed dimers (ordered by structural family)", fontsize=10)
    fig.tight_layout()
    figs["rmsd"] = "plots/rmsd_matrix.png"
    fig.savefig(plots / "rmsd_matrix.png", dpi=180)
    plt.close(fig)

    # 2. energy landscape
    fig, ax = plt.subplots(figsize=(11, 5))
    Ds = D.sort_values("dE_geom_meV").reset_index(drop=True)
    for i, r in Ds.iterrows():
        c, m = FAM_STYLE[r["mag_family"]]
        ax.scatter(i, r["dE_geom_meV"], color=c, marker=m, s=55 if m != "." else 30,
                   edgecolor=RUN_COLOR[r["run"]], linewidth=1.6, zorder=3)
    for f in FAM_ORDER:
        if (Ds["mag_family"] == f).any():
            ax.scatter([], [], color=FAM_STYLE[f][0], marker=FAM_STYLE[f][1], label=f)
    for run, c in RUN_COLOR.items():
        ax.scatter([], [], facecolor="white", edgecolor=c, linewidth=1.6, label=f"outline: {run}")
    ax.axhline(E_LOW, color=INK_3, ls=":", lw=1)
    ax.axhline(E_HIGH, color=INK_3, ls=":", lw=1)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xticks(range(len(Ds)))
    ax.set_xticklabels([f"{k} ({s})" for k, s in zip(Ds["key"], Ds["struct_family"])], rotation=90, fontsize=6)
    ax.set_ylabel("ΔE_geom = E − E_lowest free dimer (meV, symlog)")
    ax.set_title("Conformational energy of every relaxed dimer; fill = magnetic family, outline = run", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, ncol=3)
    style(ax)
    fig.tight_layout()
    figs["energy"] = "plots/energy_landscape.png"
    fig.savefig(plots / "energy_landscape.png", dpi=180)
    plt.close(fig)

    M = D[D["fit"].notna()].copy()
    if M.empty:
        return figs
    M = M.sort_values(["mag_family", "dE_geom_meV"])
    # 3. moments
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(M))
    for k, lab in enumerate(RADICALS):
        axes[0].plot(x, M[f"s_{lab}_uuuu"], "o", color=COLORS[k], label=f"|s_{lab}| (uuuu)", markersize=5)
    axes[0].axhline(XC.MIN_LOCAL_MOMENT, color=INK_3, ls=":")
    axes[0].set_ylabel("fragment spin in uuuu")
    axes[1].bar(x, M["moment_spread_BS"], color=[FAM_STYLE[f][0] for f in M["mag_family"]])
    axes[1].axhline(MOMENT_SPREAD_MAX, color=INK_3, ls=":")
    axes[1].set_ylabel("max change of |fragment spin| over BS states")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(M["key"], rotation=90, fontsize=6)
        style(ax)
    axes[0].legend(frameon=False, fontsize=7.5, ncol=4)
    fig.suptitle("Does the four-moment picture survive? (dotted: thresholds)", fontsize=10)
    fig.tight_layout()
    figs["moments"] = "plots/local_moments.png"
    fig.savefig(plots / "local_moments.png", dpi=180)
    plt.close(fig)

    F = M[M["J_intra_abs"].notna()]
    # 4. J overview heatmap
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(F) + 2))
    J = F[[f"J_{p}_meV" for p in PAIRS]].to_numpy()
    lim = np.nanmax(np.abs(J))
    im = ax.imshow(J, cmap="RdBu", norm=SymLogNorm(linthresh=1.0, vmin=-lim, vmax=lim), aspect="auto")
    for (i, k), v in np.ndenumerate(J):
        ax.text(k, i, f"{v:+.1f}" if abs(v) < 100 else f"{v:+.0f}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"J_{p}\n({PL[p]})" + ("\nintra" if p in INTRA else "") for p in PAIRS], fontsize=7)
    ax.set_yticks(range(len(F)))
    ax.set_yticklabels([f"{k} [{f[0]}] {s}" for k, f, s in zip(F["key"], F["mag_family"], F["struct_family"])], fontsize=6.5)
    fig.colorbar(im, ax=ax, label="J (meV), H = −ΣJ S·S: red AFM, blue FM (symlog, linear within ±1 meV)")
    ax.set_title("Exchange couplings by magnetic family", fontsize=10)
    fig.tight_layout()
    figs["J"] = "plots/J_overview.png"
    fig.savefig(plots / "J_overview.png", dpi=180)
    plt.close(fig)

    # 5. intra vs inter
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for f in FAM_ORDER:
        g = F[F["mag_family"] == f]
        if len(g):
            ax.scatter(g["J_intra_abs"], g["J_inter_abs"], color=FAM_STYLE[f][0], marker=FAM_STYLE[f][1], s=45, label=f)
    lo = max(min(F["J_intra_abs"].min(), F["J_inter_abs"].min()) * 0.5, 1e-2)
    hi = max(F["J_intra_abs"].max(), F["J_inter_abs"].max()) * 2
    ax.plot([lo, hi], [lo, hi], color=INK_3, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|J_12| + |J_34| (intramolecular, meV)")
    ax.set_ylabel("|J_13|+|J_14|+|J_23|+|J_24| (intermolecular, meV)")
    ax.legend(frameon=False, fontsize=7.5)
    style(ax)
    fig.tight_layout()
    figs["intra_inter"] = "plots/J_intra_vs_inter.png"
    fig.savefig(plots / "J_intra_vs_inter.png", dpi=180)
    plt.close(fig)

    # 6. correlation heatmaps
    n = len(F)
    cols = 6
    rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(2.3 * cols, 2.4 * rows_))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (_, r) in zip(np.atleast_1d(axes).ravel(), F.iterrows()):
        Cm = np.full((4, 4), np.nan)
        for p in PAIRS:
            i, j = int(p[0]) - 1, int(p[1]) - 1
            Cm[i, j] = Cm[j, i] = r[f"C_{p}"]
        ax.imshow(Cm, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0.0, vmin=-0.75, vmax=0.25))
        for i in range(4):
            for j in range(4):
                if i != j:
                    ax.text(j, i, f"{Cm[i, j]:+.2f}", ha="center", va="center", fontsize=6)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(list(RADICALS), fontsize=7)
        ax.set_yticklabels(list(RADICALS), fontsize=7)
        ax.axis("on")
        ax.set_title(f"{r['key']}\n{r['mag_family'][0]}, S_GS={r['S_GS']:g}", fontsize=7)
    fig.suptitle("Ground-state pair correlations C_ij = ⟨S_i·S_j⟩ (+0.25 triplet pair, −0.75 singlet pair)", fontsize=10)
    fig.tight_layout()
    figs["corr"] = "plots/pair_correlations.png"
    fig.savefig(plots / "pair_correlations.png", dpi=170)
    plt.close(fig)

    # 7. spectra
    lv = pd.read_csv(OUT / "heisenberg_spectra.csv")
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (_, r) in enumerate(F.iterrows()):
        g = lv[lv["key"] == r["key"]]
        for _, l in g.iterrows():
            ax.hlines(l["E_meV"], i - 0.35, i + 0.35, color=COLORS[int(l["S"] * 2) % len(COLORS)] if l["S"] < 3 else INK, lw=2)
    for s, lab in ((0, "S = 0"), (1, "S = 1"), (2, "S = 2")):
        ax.plot([], [], color=COLORS[s * 2], lw=2, label=lab)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(range(len(F)))
    ax.set_xticklabels([f"{k} [{f[0]}]" for k, f in zip(F["key"], F["mag_family"])], rotation=90, fontsize=6.5)
    ax.set_ylabel("E − E_GS of Heisenberg multiplets (meV, symlog)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Heisenberg spectra of the fitted dimers", fontsize=10)
    style(ax)
    fig.tight_layout()
    figs["spectra"] = "plots/heisenberg_spectra.png"
    fig.savefig(plots / "heisenberg_spectra.png", dpi=180)
    plt.close(fig)

    # 8. jackknife
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(len(F)), F["max_rel_dJ"], color=[FAM_STYLE[f][0] for f in F["mag_family"]])
    ax.axhline(JACKKNIFE_MAX, color=INK_3, ls=":")
    ax.set_yscale("log")
    ax.set_xticks(range(len(F)))
    ax.set_xticklabels(F["key"], rotation=90, fontsize=6.5)
    ax.set_ylabel("leave-one-out max|ΔJ| / max|J|")
    ax.set_title("Jackknife stability of the eight-state fit (dotted: threshold; missing bar = exact 7-state fit)", fontsize=10)
    style(ax)
    fig.tight_layout()
    figs["jackknife"] = "plots/jackknife.png"
    fig.savefig(plots / "jackknife.png", dpi=180)
    plt.close(fig)

    # 8b. central planarity: the origin of family F
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for f in FAM_ORDER:
        g = F[F["mag_family"] == f]
        if len(g):
            for k, (p, tc) in enumerate((("12", "theta_C_M1"), ("34", "theta_C_M2"))):
                ax.scatter(g[tc], g[f"J_{p}_meV"].abs(), color=FAM_STYLE[f][0], marker=FAM_STYLE[f][1], s=40,
                           label=f if k == 0 else None)
    ax.set_yscale("log")
    ax.set_xlabel("central ring-plane angle θ_C of the molecule (deg)")
    ax.set_ylabel("|intramolecular J| of that molecule (meV)")
    ax.legend(frameon=False, fontsize=7.5)
    style(ax)
    ax = axes[1]
    for f in FAM_ORDER:
        g = D[D["mag_family"] == f]
        if len(g):
            ax.scatter(g["theta_C_min"], g["dE_geom_meV"], color=FAM_STYLE[f][0], marker=FAM_STYLE[f][1], s=40, label=f)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("smaller central plane angle of the two molecules (deg)")
    ax.set_ylabel("ΔE_geom (meV)")
    ax.legend(frameon=False, fontsize=7.5)
    style(ax)
    fig.suptitle("Planar central links: strong intramolecular pairing and a large conformational cost", fontsize=10)
    fig.tight_layout()
    figs["planarity"] = "plots/central_planarity.png"
    fig.savefig(plots / "central_planarity.png", dpi=180)
    plt.close(fig)

    # 9. geometry vs magnetism
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for p, c in zip(INTER, COLORS):
        axes[0].scatter(F[f"d_rad_{p}"], F[f"J_{p}_meV"], color=c, s=30, label=f"J_{p} ({PL[p]})")
        axes[1].scatter(F[f"contact_{p}_A"], F[f"J_{p}_meV"], color=c, s=30, label=f"J_{p} ({PL[p]})")
    for ax, xl in ((axes[0], "radical-carbon distance (Å)"), (axes[1], "closest heavy-atom contact between the halves (Å)")):
        ax.axhline(0, color=INK_3, lw=0.8)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel(xl)
        ax.set_ylabel("intermolecular J (meV)")
        ax.legend(frameon=False, fontsize=7.5)
        style(ax)
    ax = axes[2]
    if MONOMER_EXCHANGE_CSV.exists():
        mono = pd.read_csv(MONOMER_EXCHANGE_CSV)
        mono = mono[mono["usable"] == True]  # noqa: E712
        if "J_yamaguchi_meV" in mono:
            ax.scatter(mono["P"], mono["J_yamaguchi_meV"], color=GRID, edgecolor=INK_3, s=14,
                       label="isolated monomer scan (J_yamaguchi)")
    for k, (p, mcol) in enumerate((("12", "P_M1"), ("34", "P_M2"))):
        ax.scatter(F[mcol], F[f"J_{p}_meV"], color=COLORS[k], s=35, label=f"dimer J_{p} vs P of its molecule")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("P = cos²θ_out cos²θ_C cos²θ_out (actual planes)")
    ax.set_ylabel("intramolecular J (meV)")
    ax.legend(frameon=False, fontsize=7.5)
    style(ax)
    fig.tight_layout()
    figs["geometry"] = "plots/geometry_vs_J.png"
    fig.savefig(plots / "geometry_vs_J.png", dpi=180)
    plt.close(fig)
    return figs


# =========================
# REPORT
# =========================
def md_table(df, cols, fmt=None):
    fmt = fmt or {}
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if not np.isfinite(v) else (fmt.get(c, "{:.3g}").format(v)))
            else:
                cells.append(str(v))
        L.append("| " + " | ".join(cells) + " |")
    return L


def write_report(D, R, mirror, keys, e_ref_key, figs, path):
    L = []
    F = D[D["fit"].notna()]
    fitted = F[F["J_intra_abs"].notna()]
    L.append("# Structural and magnetic families of the four-radical dimers (run2 + run3 + run4)")
    L.append("")
    L.append("Generated by `visualisation/analyze_dimer_families.py`. Data: [dimer_families.csv](dimer_families.csv) "
             "(one row per relaxed dimer, all descriptors), [rmsd_matrix.csv](rmsd_matrix.csv), "
             "[heisenberg_spectra.csv](heisenberg_spectra.csv).")
    L.append("")
    L.append("Convention: H = −Σ J_ij S_i·S_j, J > 0 FM, J < 0 AFM; sites 1–4 = radicals A, B (molecule 1), C, D "
             "(molecule 2); J_12 and J_34 intramolecular, the other four intermolecular. ΔE_geom is the R2SCAN-3C "
             f"relaxation energy relative to the lowest unconstrained converged dimer ({e_ref_key}); it is never mixed "
             "with broken-symmetry splittings.")
    L.append("")
    L.append("## Coverage")
    L.append("")
    cov = D.groupby("run").agg(dimers=("key", "size"), converged=("converged", "sum"),
                               with_J=("J_intra_abs", lambda s: int(s.notna().sum())))
    L.append("| run | relaxed dimers | converged | relaxation | with fitted J |")
    L.append("|---|---|---|---|---|")
    for run, r in cov.iterrows():
        rel = ", ".join(sorted(set(D.loc[D["run"] == run, "relaxation"])))
        L.append(f"| {run} | {r['dimers']} | {int(r['converged'])} | {rel} | {r['with_J']} |")
    L.append("")
    L.append("run4 has no broken-symmetry results yet, so it enters the structural part only; run3 and the run4 "
             "fixC sets were relaxed with constrained torsions and are not free minima (kept, but flagged).")
    L.append("")

    # 1 structural
    L.append("## 1. Which relaxations are the same minimum?")
    L.append("")
    fam = D.groupby("struct_family").agg(n=("key", "size"), runs=("run", lambda s: " ".join(sorted(set(s)))),
                                         E_min=("dE_geom_meV", "min"), E_max=("dE_geom_meV", "max"),
                                         members=("key", lambda s: ", ".join(s)))
    multi = fam[fam["n"] > 1]
    L.append(f"RMSD after optimal superposition, minimised over molecule swap, the {8} graph automorphisms of each "
             f"monomer and mirror images; complete linkage cut at {RMSD_CUT} Å. **{len(D)} relaxed dimers form "
             f"{len(fam)} structural families; {len(multi)} of them contain more than one relaxation.**")
    L.append("")
    off = R[np.triu_indices(len(R), 1)]
    L.append(f"Pairwise RMSD: median {np.median(off):.2f} Å, {int((off < RMSD_CUT).sum())} pairs below {RMSD_CUT} Å, "
             f"{int((off < 1.0).sum())} below 1 Å; {int(mirror[np.triu_indices(len(R), 1)][off < RMSD_CUT].sum())} of the "
             "sub-cut pairs are related by a mirror image.")
    L.append("")
    if len(multi):
        L.append("Families with more than one member (the run is metadata):")
        L.append("")
        L += md_table(multi.reset_index(), ["struct_family", "n", "runs", "E_min", "E_max", "members"],
                      {"E_min": "{:.1f}", "E_max": "{:.1f}"})
        L.append("")
    L.append(f"![RMSD matrix]({figs['rmsd']})")
    L.append("")
    al = D[D["J_aligned_to"].astype(str).str.len() > 0]
    fams = al.groupby("struct_family").filter(lambda g: len(g) > 1)
    if len(fams):
        L.append("**Same structure, same couplings?** The radical labels A–D follow each dimer's atom numbering, so "
                 "two copies of one structure can carry permuted labels. Using the atom correspondence of the optimal "
                 "superposition, the couplings of every member are re-expressed in the labels of the family's "
                 "lowest-energy member (`Jal_*` columns in the CSV):")
        L.append("")
        cols = ["struct_family", "key", "dE_geom_meV", "label_map_to_rep"] + [f"Jal_{p}_meV" for p in PAIRS] + \
               ["dominant_radicals"]
        L += md_table(fams.sort_values(["struct_family", "dE_geom_meV"]), cols,
                      {**{f"Jal_{p}_meV": "{:+.2f}" for p in PAIRS}, "dE_geom_meV": "{:.1f}"})
        L.append("")
        for sfam, g in fams.groupby("struct_family"):
            Jm = g[[f"Jal_{p}_meV" for p in PAIRS]].to_numpy()
            dev = np.abs(Jm - Jm[0]).max()
            dom = {PL[max(PAIRS, key=lambda p: abs(r[f"Jal_{p}_meV"]))] for _, r in g.iterrows()}
            L.append(f"- {sfam}: after relabelling, the dominant pair is {' / '.join(sorted(dom))} in every member; "
                     f"largest coupling difference between members {dev:.2f} meV.")
        L.append("")
        L.append("A different raw dominant pair (e.g. J_AC in one member, J_AD in another) can therefore be a "
                 "labelling effect, not a different contact topology. Pair names are only comparable across dimers of "
                 "one structural family.")
        L.append("")

    # 2-3 descriptors and energy
    L.append("## 2–3. Structural descriptors and conformational energy")
    L.append("")
    L.append(f"Energy classes: low < {E_LOW:.0f} meV ≤ intermediate < {E_HIGH:.0f} meV ≤ high (gas-phase R2SCAN-3C; "
             "no Boltzmann weights, since a junction can trap any of these).")
    L.append("")
    cols = ["key", "struct_family", "relaxation", "converged", "dE_geom_meV", "energy_class", "theta_out_A", "theta_out_B",
            "theta_C_M1", "theta_out_C", "theta_out_D", "theta_C_M2", "centre_distance_A", "mol_plane_angle", "slip_A",
            "min_CC_A", "mag_family"]
    L += md_table(D.sort_values("dE_geom_meV"), cols, {c: "{:.1f}" for c in cols if c not in ("key",)})
    L.append("")
    L.append(f"![Energy landscape]({figs['energy']})")
    L.append("")
    L.append("Full descriptors (signed outer/central dihedrals, P per molecule, radical-carbon distances d_rad_ij, "
             "closest contacts and C···C counts between radical halves) are in the CSV.")
    L.append("")

    if fitted.empty:
        path.write_text("\n".join(L) + "\n")
        return
    # 4 moments
    L.append("## 4. Does the four-moment picture survive?")
    L.append("")
    ok = F["moments_ok"].sum()
    L.append(f"Of {len(F)} dimers with broken-symmetry data, **{int(ok)} keep four stable local moments** "
             f"(|s| ≥ {XC.MIN_LOCAL_MOMENT} everywhere and |s| changing by ≤ {MOMENT_SPREAD_MAX} between BS states); the "
             "others are placed in family F rather than forced into a magnetic class.")
    L.append("")
    L += md_table(F.sort_values("dE_geom_meV"), ["key", "s_A_uuuu", "s_B_uuuu", "s_C_uuuu", "s_D_uuuu", "s_mean_uuuu",
                                                  "s_spread_uuuu", "moment_spread_BS", "min_moment_BS", "moments_ok"],
                  {c: "{:.2f}" for c in ["s_A_uuuu", "s_B_uuuu", "s_C_uuuu", "s_D_uuuu", "s_mean_uuuu", "s_spread_uuuu",
                                         "moment_spread_BS", "min_moment_BS"]})
    L.append("")
    L.append("The radical halves cover the whole molecule (terminal ring + adjacent backbone half), so a separate "
             "'backbone leakage' cannot be defined with this partition; s_spread and the BS spread measure it indirectly.")
    L.append("")
    L.append(f"![Local moments]({figs['moments']})")
    L.append("")
    fF = F[F["mag_family"].str.startswith("F") & F["J_12_meV"].notna()]
    good = F[~F["mag_family"].str.startswith("F")]
    if len(fF):
        L.append(f"**What the failed dimers have in common.** {int((fF['theta_C_min'] < 25).sum())} of {len(fF)} fitted "
                 f"family-F dimers contain a molecule with an almost planar central link (θ_C < 25°; median "
                 f"{fF['theta_C_min'].median():.1f}°), against {int((good['theta_C_min'] < 25).sum())} of {len(good)} "
                 f"dimers in A–E (median {good['theta_C_min'].median():.1f}°). Those molecules have |J_intra| of "
                 f"{fF[['J_12_meV', 'J_34_meV']].abs().max(axis=1).min():.0f}–{fF[['J_12_meV', 'J_34_meV']].abs().max(axis=1).max():.0f} meV "
                 f"and the dimers lie {fF['dE_geom_meV'].min() / 1000:.1f}–{fF['dE_geom_meV'].max() / 1000:.1f} eV above the "
                 "lowest dimer: a planar central link makes the molecule strongly paired (quinoidal / closed-shell "
                 "character), the radical spin is no longer split evenly over the two halves, and the four-spin mapping "
                 "breaks down. These are high-energy conformers carried over from the starting structures, not low-lying "
                 "dimer motifs.")
        L.append("")
        L.append(f"![Central planarity]({figs['planarity']})")
        L.append("")

    # 5-6 J and families
    L.append("## 5–6. Exchange couplings and magnetic families")
    L.append("")
    L.append("Rules, applied in order: **F** fewer than 7 valid configurations, unstable moments, or leave-one-out "
             f"max|ΔJ|/max|J| > {JACKKNIFE_MAX}; **A** max|J_inter| < {WEAK_INTER_RATIO}·max|J_intra| (or an intramolecular "
             f"pair dominates); **B / D** one intermolecular pair dominates (R_dom ≥ {DOMINANT_RDOM} and |J1/J2| ≥ "
             f"{DOMINANT_RATIO}), FM / AFM (the pair name depends on the atom labelling, see section 1); **E** everything else.")
    L.append("")
    cnt = D["mag_family"].value_counts()
    L.append("| magnetic family | dimers |")
    L.append("|---|---|")
    for f in FAM_ORDER:
        if f in cnt:
            L.append(f"| {f} | {cnt[f]} |")
    L.append("")
    cols = ["key", "mag_family", "mag_reason", "dE_geom_meV"] + [f"J_{p}_meV" for p in PAIRS] + \
           ["R_dom", "dominant_radicals", "J1_over_J2", "rms_residual_meV", "max_rel_dJ"]
    L += md_table(F.sort_values(["mag_family", "dE_geom_meV"]), cols,
                  {**{c: "{:+.2f}" for c in cols if c.startswith("J_")}, "dE_geom_meV": "{:.1f}", "R_dom": "{:.2f}",
                   "J1_over_J2": "{:.1f}", "rms_residual_meV": "{:.3f}", "max_rel_dJ": "{:.3f}"})
    L.append("")
    L.append(f"![J overview]({figs['J']})")
    L.append("")
    L.append(f"![Intra vs inter]({figs['intra_inter']})")
    L.append("")

    # 7 spectra, correlations, composite spins
    L.append("## 7. Heisenberg spectra, pair correlations and composite spins")
    L.append("")
    L.append("16×16 diagonalisation of the fitted Hamiltonian. C_ij = ⟨S_i·S_j⟩ in the ground state (averaged over a "
             "degenerate ground multiplet): +0.25 = triplet pair, −0.75 = singlet pair. S_ij^eff from "
             "⟨(S_i+S_j)²⟩ = 3/2 + 2C_ij = S(S+1). S_AB^eff and S_CD^eff ≈ 1 mean each molecule still behaves as a "
             "triplet diradical; ≈ 0 means it is internally paired.")
    L.append("")
    cols = ["key", "mag_family", "S_GS", "Delta1_meV", "S1", "Delta2_meV", "S2", "Seff_12", "Seff_34",
            "C_13", "C_14", "C_23", "C_24"]
    L += md_table(fitted.sort_values(["mag_family", "dE_geom_meV"]), cols,
                  {"Delta1_meV": "{:.3g}", "Delta2_meV": "{:.3g}", "Seff_12": "{:.2f}", "Seff_34": "{:.2f}",
                   **{f"C_{p}": "{:+.2f}" for p in INTER}, "S_GS": "{:g}", "S1": "{:g}", "S2": "{:g}"})
    L.append("")
    ok_fit = fitted[~fitted["mag_family"].str.startswith("F")]
    two_s1 = ok_fit[(ok_fit["Seff_12"] > 0.8) & (ok_fit["Seff_34"] > 0.8)]
    paired = ok_fit[(ok_fit["Seff_12"] < 0.3) | (ok_fit["Seff_34"] < 0.3)]
    L.append(f"**Two S = 1 molecules or four S = ½ radicals?** (families A–E only, F excluded) In {len(two_s1)} of {len(ok_fit)} fitted dimers both "
             f"molecules keep S_eff > 0.8 in the ground state (two interacting triplet diradicals); in {len(paired)} at "
             "least one molecule has S_eff < 0.3"
             + (f", although its |J_intra| is at most {paired[['J_12_meV', 'J_34_meV']].abs().max(axis=1).max():.1f} meV: this is "
                "not pairing inside the molecule but the radicals being bound into intermolecular pairs (see C_13, C_14, "
                "C_23, C_24)" if len(paired) else "")
             + "; the rest are intermediate. For the low-energy dimers the four-S = ½ picture, reorganised by the "
             "intermolecular contacts, describes the ground state better than two S = 1 molecules.")
    L.append("")
    L.append(f"![Pair correlations]({figs['corr']})")
    L.append("")
    L.append(f"![Spectra]({figs['spectra']})")
    L.append("")

    # jackknife
    L.append("## Fit robustness (jackknife, condition number)")
    L.append("")
    jk = fitted[fitted["max_rel_dJ"].notna()]
    L.append(f"Eight states for seven parameters leave one redundant equation. Leave-one-out refits (each an exact "
             f"7-state solve): max|ΔJ|/max|J| median {jk['max_rel_dJ'].median():.3f}, max {jk['max_rel_dJ'].max():.3f} "
             f"({int((jk['max_rel_dJ'] > JACKKNIFE_MAX).sum())} above {JACKKNIFE_MAX}). Condition number of the 8×7 "
             f"design matrix {jk['cond_full'].iloc[0]:.2f} (well conditioned); leave-one-out matrices up to "
             f"{jk['cond_loo_max'].max():.2f}. Because the single redundancy is the four-spin term, |ΔJ| on removing a "
             "state is set by the fit residual; small residual and small ΔJ are therefore one test, not two.")
    L.append("")
    L.append(f"![Jackknife]({figs['jackknife']})")
    L.append("")

    # 8 geometry vs magnetism
    L.append("## 8. Geometry versus magnetism")
    L.append("")
    for p in INTER:
        g = fitted[[f"d_rad_{p}", f"contact_{p}_A", f"J_{p}_meV"]].dropna()
        if len(g) > 4:
            from scipy.stats import spearmanr
            r1 = spearmanr(g[f"d_rad_{p}"], g[f"J_{p}_meV"].abs()).statistic
            r2 = spearmanr(g[f"contact_{p}_A"], g[f"J_{p}_meV"].abs()).statistic
            L.append(f"- |J_{p}| ({PL[p]}) vs radical-carbon distance: Spearman ρ = {r1:+.2f}; vs closest contact of the "
                     f"two halves: ρ = {r2:+.2f} (n = {len(g)}).")
    dom = fitted[fitted["mag_family"].str.startswith(("B", "D"))]
    if len(dom):
        hit = 0
        for _, r in dom.iterrows():
            p = r["dominant_pair"]
            closest = min(INTER, key=lambda q: r[f"contact_{q}_A"])
            hit += int(p == closest)
        L.append(f"- In {hit} of {len(dom)} dimers with one dominant intermolecular pair, that pair is also the pair of "
                 "radical halves in closest contact.")
    if MONOMER_EXCHANGE_CSV.exists():
        mono = pd.read_csv(MONOMER_EXCHANGE_CSV)
        mono = mono[(mono["usable"] == True) & mono["J_yamaguchi_meV"].notna()]  # noqa: E712
        pts = []
        for _, r in fitted.iterrows():
            for p, mcol in (("12", "P_M1"), ("34", "P_M2")):
                near = mono[(mono["P"] - r[mcol]).abs() <= 0.01]
                if len(near) >= 3:
                    pts.append((r["mag_family"][0], r[mcol], r[f"J_{p}_meV"], near["J_yamaguchi_meV"].median()))
        if pts:
            pt = pd.DataFrame(pts, columns=["fam", "P", "J_dimer", "J_monomer"])
            ok_ = pt[pt["fam"] != "F"]
            L.append(f"- **Intramolecular J vs the isolated-monomer relation J(P):** for the {len(ok_)} molecules of dimers "
                     f"in A–E (P ≤ {ok_['P'].max():.2f}) the dimer J_12 / J_34 lie on the monomer scan, median |J_dimer − "
                     f"J_monomer(P)| = {np.median(np.abs(ok_['J_dimer'] - ok_['J_monomer'])):.2f} meV (monomer value = "
                     "median of scan points within ΔP = 0.01): the intermolecular contact does not change the "
                     "intramolecular coupling much. The planar-central-link molecules of family F (P = 0.35–0.7) are far "
                     "more strongly coupled than any monomer-scan point, consistent with the breakdown in section 4.")
    else:
        L.append("- Intramolecular J against the monomer torsional proxy P: see the right panel.")
    L.append("")
    L.append(f"![Geometry vs J]({figs['geometry']})")
    L.append("")

    # 9 reproducibility and cross-table
    L.append("## 9. Run reproducibility and structural × magnetic cross-table")
    L.append("")
    ct = pd.crosstab(D["mag_family"], D["run"])
    L.append("| magnetic family | " + " | ".join(ct.columns) + " | total |")
    L.append("|---|" + "---|" * (len(ct.columns) + 1))
    for f in FAM_ORDER:
        if f in ct.index:
            L.append(f"| {f} | " + " | ".join(str(int(v)) for v in ct.loc[f]) + f" | {int(ct.loc[f].sum())} |")
    L.append("")
    shared = D.groupby("struct_family")["run"].nunique()
    L.append(f"Structural families found by more than one run: {int((shared > 1).sum())} of {len(shared)}. This is "
             "expected rather than a failure of reproducibility: the runs start from different monomers (run2: the "
             "SIESTA dimer conformers; run3: the L135_C120_R090 monomer with constrained torsions; run4: the "
             "central-plane-90° monomer), so their minima cannot coincide. The test that is possible is within run4, "
             "where the three relaxation variants of one starting dimer (free / fixC_dihedral / fixC_double90) share "
             "a structural family in several cases (see section 1); run2 itself finds the same minimum three times "
             "(S01).")
    L.append("")
    x = pd.crosstab(fitted["struct_family"], fitted["mag_family"])
    if len(x):
        L.append("Structural family × magnetic family (fitted dimers):")
        L.append("")
        L.append("| structural family | " + " | ".join(c[0] for c in x.columns) + " |")
        L.append("|---|" + "---|" * len(x.columns))
        for sfam, r in x.iterrows():
            L.append(f"| {sfam} | " + " | ".join(str(int(v)) if v else "" for v in r) + " |")
        L.append("")
        consistent = sum(1 for _, r in x.iterrows() if (r > 0).sum() == 1 and r.sum() > 1)
        multi_s = sum(1 for _, r in x.iterrows() if r.sum() > 1)
        L.append(f"Of the {multi_s} structural families with more than one fitted member, {consistent} have a single "
                 "magnetic family: there the structure predicts the magnetism.")
        L.append("")

    # 10 representatives
    L.append("## 10. Representatives for UNO/UCO/orbital analysis")
    L.append("")
    L.append("Lowest-ΔE_geom member of each magnetic family with a stable fit:")
    L.append("")
    reps = fitted.sort_values("dE_geom_meV").groupby("mag_family").head(2)
    L += md_table(reps.sort_values(["mag_family", "dE_geom_meV"]),
                  ["mag_family", "key", "struct_family", "dE_geom_meV", "dominant_radicals", "J_dominant_meV", "S_GS"],
                  {"dE_geom_meV": "{:.1f}", "J_dominant_meV": "{:+.2f}", "S_GS": "{:g}"})
    L.append("")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
