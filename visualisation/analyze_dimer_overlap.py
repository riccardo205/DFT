#!/usr/bin/env python3
"""
Structural overlap and magnetic-orbital overlap between the radical fragments of the dimers_run2
four-radical dimers, compared with the pairwise exchange J_ij.

Fragments (configurable, USER SETTINGS). Every radical half (A, B on molecule 1; C, D on molecule 2,
read from <run>_broken_symmetry/.../radical_fragments.json, the same atoms used for the FlipSpin
inputs and the J fit) consists of two rigid conjugated units, defined in the reference monomer
numbering (L135_C120_R090, mapped onto every molecule by graph isomorphism):
    fluorenyl : 13 C, 5-ring + two benzo rings, contains the radical carbon C9 (ref 28 / 41)
    anthryl   : 14 C, three linearly fused 6-rings; bonded to C9 of its fluorenyl and, through the
                central bond 9-17, to the anthryl of the other half
PLANE_UNIT selects the unit used for the pair descriptors (default fluorenyl, the radical
aromatic fragment); "half_carbons" uses all 27 C of the half, "custom" uses CUSTOM_REF_UNITS.

Pair descriptors (unit P on the first radical of the pair, Q on the second):
    rms_plane_P/Q       RMS distance of the unit atoms from their own SVD best plane (planarity)
    tilt_deg            angle between the plane normals (0 = face-to-face parallel)
    sep_normal_A        |(c_Q - c_P) . n_mean|, centroid separation along the mean normal
    interplanar_A       mean point-to-plane distance of each unit's atoms from the other plane
    slip_A              lateral centroid offset (component of c_Q - c_P perpendicular to n_mean)
    slip_short_A        slip along P's in-plane C2 axis v_P (signed: > 0 = towards P's C9)
    slip_long_A         |slip| along P's in-plane long axis u_P = n_mean x v_P (sign is not defined
                        because the two benzo rings are symmetry-equivalent)
    phi_C2_deg          in-plane rotation: angle between v_P and v_Q projected on the mean plane,
                        0..180 (0 = both C9 point the same way, 180 = head-to-tail)
    chi_long_deg        angle between the long axes, min(phi, 180 - phi), 0..90
    c9_lateral_A, d_C9_C9_A   lateral offset / distance of the two radical carbons
    overlap_area_A2     area of overlap of the projected ring polygons on the mean plane
    contact_*           closest atom-atom contact between the two halves (and between the units)
    n_CC_<cut>          C...C pairs between the halves closer than CONTACT_CUT
    mcconnell           sum over C...C contacts < MC_CUT of rho_a rho_b (HS Mulliken spins)
    S2_mag, S_mag       squared / absolute overlap of the magnetic orbitals (from the UCO overlaps)
The in-plane axes are built from atom coordinates only: v = projection of (C9 - unit centroid) on
the mean plane (the fluorenyl C2 axis, polar), u = n_mean x v. Plane normals alone only measure
the tilt; v/u resolve the in-plane rotation and split the slip into two directions.

Magnetic-orbital overlap from the UHF corresponding orbitals (UCO, printed by ORCA for every BS
run): for a BS state with k flipped radicals, the k smallest corresponding-pair overlaps T are the
overlaps between the beta magnetic orbitals of the flipped radicals and the alpha magnetic space
of the others. With x_ij = S_ij^2 (magnetic orbitals assumed transferable between configurations,
as in the Ising fit): single flip of i -> T_i^2 = sum_j x_ij; pair flip (i,j | k,l) -> T_1^2 + T_2^2
= x_ik + x_il + x_jk + x_jl. 7 equations, 6 unknowns, solved by least squares (one closure
residual). Only configurations with used_in_fit (pattern_ok and local_moments_ok) are used.

Also: internal geometry of each molecule (angles between the unit planes, fluorenyl butterfly,
anthryl bend, planarity) and the spin distribution of each half over fluorenyl / anthryl / H.

Inputs: <run>_broken_symmetry trees (geometry, radical_fragments.json, .out), bs_energies.csv and
exchange_couplings.csv (Read_output_bs_orca.py), families/dimer_families.csv (analyze_dimer_families.py).
Outputs: visualisation/output/orca/dimers_run2/overlap/ (overlap_analysis.md, CSVs, plots/).
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
from matplotlib.path import Path as MPath
from networkx.algorithms.isomorphism import GraphMatcher

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "orca"))
import exchange_convention as XC  # noqa: E402
from analyze_dimer_families import COLORS, GRID, INK, INK_2, INK_3, natural_key, style  # noqa: E402
from prepare_dimers_broken_symmetry import bond_graph, mulliken_spins  # noqa: E402

# =========================
# USER SETTINGS
# =========================
DIMERS_ROOT = REPO_ROOT / "runs" / "orca" / "short-2OS" / "dimers_run2"
CSV_ROOT = REPO_ROOT / "visualisation" / "output" / "orca" / "dimers_run2"
OUT = CSV_ROOT / "overlap"
REF_XYZ = REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans" / "angle_scan_relaxation" / "L135_C120_R090" / "high_spin.xyz"
RUNS = ["run2", "run3", "run4"]              # runs without exchange_couplings.csv are skipped
STAGE = "relaxed"

PLANE_UNIT = "fluorenyl"                     # "fluorenyl" | "anthryl" | "half_carbons" | "custom"
# units in reference numbering, keyed by the reference radical carbon of the half (28 or 41)
REF_UNITS = {
    "fluorenyl": {28: list(range(28, 41)), 41: list(range(41, 54))},
    "anthryl": {28: list(range(14, 28)), 41: list(range(0, 14))},
}
CUSTOM_REF_UNITS = {28: list(range(28, 41)), 41: list(range(41, 54))}   # used when PLANE_UNIT == "custom"
REF_RADICAL_C = (28, 41)

CONTACT_CUT = 4.0            # A, C...C contacts between halves
MC_CUT = 4.0                 # A, C...C contacts entering the McConnell sum
GRID_STEP = 0.05             # A, raster step for the projected overlap area
TILT_WARN = 30.0             # deg, above this phi / slip decomposition are only indicative
RELIABLE_MAX_REL_DJ = 0.25   # jackknife limit (same as analyze_dimer_families.JACKKNIFE_MAX)
# (key, pair) for the projection figure; None = automatic (strongest AFM / FM dominant inter pairs)
SELECTED = None
N_SELECT_EACH = 3

RADICALS = "ABCD"
PAIRS = XC.pair_labels(4)
PL = {p: RADICALS[int(p[0]) - 1] + RADICALS[int(p[1]) - 1] for p in PAIRS}
INTRA = {"12", "34"}
INTER = [p for p in PAIRS if p not in INTRA]
CONFIGS = ["uuuu", "duuu", "uduu", "uudu", "uuud", "dduu", "dudu", "duud"]
FM_C, AFM_C, P_C, Q_C = COLORS[7], COLORS[0], COLORS[0], COLORS[1]


# =========================
# REFERENCE TOPOLOGY
# =========================
def cycle_order(ring, g):
    """Atoms of a ring in cyclic order (walk along the ring bonds)."""
    ring = set(ring)
    order = [min(ring)]
    while len(order) < len(ring):
        nxt = [n for n in g[order[-1]] if n in ring and n not in order]
        order.append(min(nxt))
    return order


class Reference:
    def __init__(self, xyz):
        ref = read(xyz)
        self.n = len(ref)
        self.z = ref.numbers
        self.graph = bond_graph(ref)
        cg = self.graph.subgraph([i for i in self.graph if self.z[i] == 6])
        self.rings = [cycle_order(r, cg) for r in nx.minimum_cycle_basis(cg)]
        # check the hard-coded units against the ring topology: fused carbon ring systems
        fused = nx.Graph()
        fused.add_nodes_from(range(len(self.rings)))
        fused.add_edges_from((i, j) for i, j in combinations(range(len(self.rings)), 2)
                             if len(set(self.rings[i]) & set(self.rings[j])) >= 2)
        systems = {frozenset().union(*[set(self.rings[k]) for k in c]) for c in nx.connected_components(fused)}
        for unit in ("fluorenyl", "anthryl"):
            for rc in REF_RADICAL_C:
                if frozenset(REF_UNITS[unit][rc]) not in systems:
                    raise ValueError(f"REF_UNITS[{unit}][{rc}] is not a fused ring system of the reference")
        # fluorenyl sub-rings: 5-ring (contains C9) and the two benzo rings
        self.fl_rings = {}
        for rc in REF_RADICAL_C:
            fl = set(REF_UNITS["fluorenyl"][rc])
            rs = [r for r in self.rings if set(r) <= fl]
            five = [r for r in rs if rc in r]
            self.fl_rings[rc] = {"five": five[0], "benzo": [r for r in rs if rc not in r]}
        self.an_rings = {}
        for rc in REF_RADICAL_C:
            an = set(REF_UNITS["anthryl"][rc])
            rs = [r for r in self.rings if set(r) <= an]
            central_atom = 9 if 9 in an else 17
            mid = [r for r in rs if central_atom in r]
            self.an_rings[rc] = {"mid": mid[0], "outer": [r for r in rs if central_atom not in r]}

    def map_monomer(self, mono):
        gm = GraphMatcher(self.graph, bond_graph(mono), node_match=lambda a, b: a["z"] == b["z"])
        if not gm.is_isomorphic():
            raise ValueError("monomer not isomorphic to the reference")
        return gm.mapping


# =========================
# GEOMETRY HELPERS
# =========================
def fit_plane(x):
    c = x.mean(0)
    n = np.linalg.svd(x - c)[2][-1]
    return c, n, float(np.sqrt((((x - c) @ n) ** 2).mean()))


def plane_angle(n1, n2):
    return float(np.degrees(np.arccos(np.clip(abs(n1 @ n2), 0, 1))))


def unit_vec(v):
    return v / np.linalg.norm(v)


def raster_area(polys_p, polys_q):
    """Area of the union of polys_p, of polys_q and of their intersection (2D polygons)."""
    allp = np.vstack(polys_p + polys_q)
    lo, hi = allp.min(0) - 0.2, allp.max(0) + 0.2
    xs = np.arange(lo[0], hi[0], GRID_STEP)
    ys = np.arange(lo[1], hi[1], GRID_STEP)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])

    def mask(polys):
        m = np.zeros(len(pts), bool)
        for p in polys:
            m |= MPath(p).contains_points(pts)
        return m
    mp, mq = mask(polys_p), mask(polys_q)
    a = GRID_STEP ** 2
    return mp.sum() * a, mq.sum() * a, (mp & mq).sum() * a


# =========================
# ORCA OUTPUT PARSING
# =========================
def magnetic_overlaps(text, k):
    """k smallest corresponding-orbital overlaps of the last UCO block among the N_beta matched pairs."""
    blocks = re.findall(r"Orbital\s+Overlap\(\*\)\s*\n-+\n(.*?)\n\(\*\)", text, re.S)
    nb = re.findall(r"N\(Beta\)\s*:\s*(-?\d+\.\d+)", text)
    if not blocks or not nb:
        return None
    n_beta = int(round(float(nb[-1])))
    vals = [(int(i), abs(float(v))) for i, v in re.findall(r"^\s*(\d+):\s+(-?\d+\.\d+)", blocks[-1], re.M)]
    vals = sorted(v for i, v in vals if i < n_beta)
    return vals[:k]


def overlap_decomposition(T2):
    """T2: {config: [T^2 of its magnetic pairs]} (valid configs only). Least squares for x_ij = S_ij^2."""
    rows, rhs = [], []
    for c, t in T2.items():
        down = [i for i, s in enumerate(c) if s == "d"]
        up = [i for i, s in enumerate(c) if s == "u"]
        if not down:
            continue
        row = np.zeros(len(PAIRS))
        for i in down:
            for j in up:
                row[PAIRS.index("".join(map(str, sorted((i + 1, j + 1)))))] = 1
        rows.append(row)
        rhs.append(sum(t))
    out = {"S2": {p: np.nan for p in PAIRS}, "n_eq": len(rows), "closure": np.nan}
    if not rows:
        return out
    A, b = np.array(rows), np.array(rhs)
    if np.linalg.matrix_rank(A) < len(PAIRS):
        return out
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    out["S2"] = dict(zip(PAIRS, x))
    out["closure"] = float(np.sqrt(((A @ x - b) ** 2).sum())) if len(rows) > len(PAIRS) else 0.0
    return out


# =========================
# PER-DIMER ANALYSIS
# =========================
def load_inputs():
    J = [pd.read_csv(CSV_ROOT / f"{r}_broken_symmetry" / "exchange_couplings.csv").assign(run=r)
         for r in RUNS if (CSV_ROOT / f"{r}_broken_symmetry" / "exchange_couplings.csv").exists()]
    E = [pd.read_csv(CSV_ROOT / f"{r}_broken_symmetry" / "bs_energies.csv").assign(run=r)
         for r in RUNS if (CSV_ROOT / f"{r}_broken_symmetry" / "bs_energies.csv").exists()]
    J, E = pd.concat(J), pd.concat(E)
    J, E = J[J["stage"] == STAGE], E[E["stage"] == STAGE]
    fam_csv = CSV_ROOT / "families" / "dimer_families.csv"
    F = pd.read_csv(fam_csv) if fam_csv.exists() else pd.DataFrame(columns=["key"])
    return J, E, F


def analyse_dimer(run, dimer, ref, e_rows):
    d = DIMERS_ROOT / f"{run}_broken_symmetry" / STAGE / dimer
    atoms = read(d / "uuuu" / "geometry.xyz")
    pos, z, n = atoms.positions, atoms.numbers, ref.n
    halves = {k: v for k, v in json.loads((d / "radical_fragments.json").read_text())["fragments"].items()}
    # map reference units onto every half
    units, rc_of, ref_rc = {}, {}, {}
    for m in (0, 1):
        mp = {r: i + m * n for r, i in ref.map_monomer(atoms[m * n:(m + 1) * n]).items()}
        for lab in RADICALS[2 * m:2 * m + 2]:
            h = set(halves[lab])
            rcs = [rc for rc in REF_RADICAL_C if mp[rc] in h]
            if len(rcs) != 1:
                raise ValueError(f"{run}/{dimer} {lab}: {len(rcs)} radical carbons in the half")
            rc = rcs[0]
            ref_rc[lab], rc_of[lab] = rc, mp[rc]
            u = {name: [mp[i] for i in REF_UNITS[name][rc]] for name in REF_UNITS}
            u["custom"] = [mp[i] for i in CUSTOM_REF_UNITS[rc]]
            u["half_carbons"] = [i for i in halves[lab] if z[i] == 6]
            u["H"] = [i for i in halves[lab] if z[i] == 1]
            u["fl_five"] = [mp[i] for i in ref.fl_rings[rc]["five"]]
            u["fl_benzo"] = [[mp[i] for i in r] for r in ref.fl_rings[rc]["benzo"]]
            u["an_mid"] = [mp[i] for i in ref.an_rings[rc]["mid"]]
            u["an_outer"] = [[mp[i] for i in r] for r in ref.an_rings[rc]["outer"]]
            u["rings_fl"] = [u["fl_five"]] + u["fl_benzo"]
            u["rings_an"] = [u["an_mid"]] + u["an_outer"]
            for name in ("fluorenyl", "anthryl"):
                if not set(u[name]) <= h:
                    raise ValueError(f"{run}/{dimer} {lab}: {name} not inside the radical half")
            units[lab] = u
    if PLANE_UNIT in ("half_carbons", "custom", "fluorenyl", "anthryl"):
        sel = {lab: units[lab][PLANE_UNIT] for lab in RADICALS}
    else:
        raise ValueError(f"unknown PLANE_UNIT {PLANE_UNIT}")
    ring_key = "rings_an" if PLANE_UNIT == "anthryl" else "rings_fl"

    # ---- ORCA outputs: spins and magnetic overlaps
    spins, T2, spin_rows = {}, {}, []
    ok = {r["config"]: bool(r["used_in_fit"]) for _, r in e_rows.iterrows()}
    for c in CONFIGS:
        out = d / c / f"{c}.out"
        if not out.exists():
            continue
        text = out.read_text(errors="ignore")
        if "ORCA TERMINATED NORMALLY" not in text:
            continue
        sp = mulliken_spins(out)
        rho = np.array([sp.get(i, 0.0) for i in range(len(atoms))])
        spins[c] = rho
        k = c.count("d")
        if k and ok.get(c, False):
            t = magnetic_overlaps(text, k)
            if t is not None and len(t) == k:
                T2[c] = [v ** 2 for v in t]
        for lab in RADICALS:
            u = units[lab]
            tot = rho[halves[lab]].sum()
            spin_rows.append({"config": c, "used_in_fit": ok.get(c, False), "radical": lab, "spin_half": tot,
                              "spin_fluorenyl": rho[u["fluorenyl"]].sum(), "spin_anthryl": rho[u["anthryl"]].sum(),
                              "spin_H": rho[u["H"]].sum(), "spin_C9": rho[rc_of[lab]],
                              "frac_fluorenyl": rho[u["fluorenyl"]].sum() / tot if abs(tot) > 1e-6 else np.nan,
                              "abs_spin_half": np.abs(rho[halves[lab]]).sum()})
    dec = overlap_decomposition(T2)
    rho_hs = spins.get("uuuu")

    # ---- internal geometry per molecule
    internal = []
    for m, (a, b) in enumerate((("A", "B"), ("C", "D"))):
        P = {}
        for lab in (a, b):
            u = units[lab]
            P[f"fl_{lab}"] = fit_plane(pos[u["fluorenyl"]])
            P[f"an_{lab}"] = fit_plane(pos[u["anthryl"]])
            bz = [fit_plane(pos[r])[1] for r in u["fl_benzo"]]
            ao = [fit_plane(pos[r])[1] for r in u["an_outer"]]
            internal.append({"molecule": f"M{m + 1}", "radical": lab,
                             "theta_fl_an_deg": plane_angle(P[f"fl_{lab}"][1], P[f"an_{lab}"][1]),
                             "fl_butterfly_deg": plane_angle(*bz), "an_bend_deg": plane_angle(*ao),
                             "rms_fluorenyl_A": P[f"fl_{lab}"][2], "rms_anthryl_A": P[f"an_{lab}"][2]})
        for row in internal[-2:]:
            row["theta_an_an_deg"] = plane_angle(P[f"an_{a}"][1], P[f"an_{b}"][1])
            row["theta_fl_fl_deg"] = plane_angle(P[f"fl_{a}"][1], P[f"fl_{b}"][1])

    # ---- pair descriptors
    pairs = []
    for p in PAIRS:
        a, b = PL[p]
        ia, ib = sel[a], sel[b]
        cP, nP, rP = fit_plane(pos[ia])
        cQ, nQ, rQ = fit_plane(pos[ib])
        if nP @ nQ < 0:
            nQ = -nQ
        nm = unit_vec(nP + nQ)
        dv = cQ - cP
        dperp = dv - (dv @ nm) * nm
        vP = unit_vec((pos[rc_of[a]] - cP) - ((pos[rc_of[a]] - cP) @ nm) * nm)
        vQ = unit_vec((pos[rc_of[b]] - cQ) - ((pos[rc_of[b]] - cQ) @ nm) * nm)
        uP = np.cross(nm, vP)
        phi = float(np.degrees(np.arccos(np.clip(vP @ vQ, -1, 1))))
        c9 = pos[rc_of[b]] - pos[rc_of[a]]
        # projected overlap area of the ring polygons (2D frame u_P, v_P)
        to2d = lambda x: np.column_stack([(x - cP) @ uP, (x - cP) @ vP])  # noqa: E731
        ar_p, ar_q, ar_pq = raster_area([to2d(pos[r]) for r in units[a][ring_key]],
                                        [to2d(pos[r]) for r in units[b][ring_key]])
        # contacts
        ha, hb = halves[a], halves[b]
        dd = np.linalg.norm(pos[ha][:, None] - pos[hb][None], axis=-1)
        i, j = np.unravel_index(dd.argmin(), dd.shape)
        du = np.linalg.norm(pos[ia][:, None] - pos[ib][None], axis=-1)
        k1, k2 = np.unravel_index(du.argmin(), du.shape)
        ca, cb = [x for x in ha if z[x] == 6], [x for x in hb if z[x] == 6]
        dcc = np.linalg.norm(pos[ca][:, None] - pos[cb][None], axis=-1)
        row = {"pair": p, "radicals": a + b, "relation": "intra" if p in INTRA else "inter",
               "unit": PLANE_UNIT, "rms_plane_P_A": rP, "rms_plane_Q_A": rQ,
               "tilt_deg": plane_angle(nP, nQ), "sep_normal_A": abs(dv @ nm),
               "interplanar_A": 0.5 * (np.abs((pos[ib] - cP) @ nP).mean() + np.abs((pos[ia] - cQ) @ nQ).mean()),
               "centroid_distance_A": float(np.linalg.norm(dv)), "slip_A": float(np.linalg.norm(dperp)),
               "slip_short_A": float(dperp @ vP), "slip_long_A": float(abs(dperp @ uP)),
               "phi_C2_deg": phi, "chi_long_deg": min(phi, 180 - phi),
               "d_C9_C9_A": float(np.linalg.norm(c9)), "c9_lateral_A": float(np.linalg.norm(c9 - (c9 @ nm) * nm)),
               "area_P_A2": ar_p, "area_Q_A2": ar_q, "overlap_area_A2": ar_pq,
               "overlap_frac": ar_pq / min(ar_p, ar_q),
               "contact_half_A": float(dd[i, j]), "contact_half_atoms": f"{atom_label(ha[i], z, units[a], rc_of[a])}-"
                                                                      f"{atom_label(hb[j], z, units[b], rc_of[b])}",
               "contact_unit_A": float(du[k1, k2]), "contact_unit_atoms": f"{atom_label(ia[k1], z, units[a], rc_of[a])}-"
                                                                      f"{atom_label(ib[k2], z, units[b], rc_of[b])}",
               f"n_CC_{CONTACT_CUT:.1f}": int((dcc < CONTACT_CUT).sum())}
        if rho_hs is not None:
            close = dcc < MC_CUT
            prod = np.outer(rho_hs[ca], rho_hs[cb])
            row["mcconnell"] = float(prod[close].sum())
            row["mcconnell_pos"] = float(prod[close & (prod > 0)].sum())
            row["mcconnell_neg"] = float(prod[close & (prod < 0)].sum())
        if p in INTRA:   # the two halves are bonded (central bond 9-17): no through-space contact
            for c in ("contact_half_A", "contact_half_atoms", f"n_CC_{CONTACT_CUT:.1f}", "mcconnell",
                      "mcconnell_pos", "mcconnell_neg"):
                row[c] = np.nan
        row["S2_mag"] = dec["S2"][p]
        row["S_mag"] = float(np.sqrt(max(dec["S2"][p], 0))) if np.isfinite(dec["S2"][p]) else np.nan
        row["S_n_eq"], row["S_closure"] = dec["n_eq"], dec["closure"]
        pairs.append(row)
    geo = {"pos": pos, "z": z, "units": units, "halves": halves, "rc": rc_of, "rho": rho_hs, "sel": sel,
           "ring_key": ring_key, "T2": T2}
    return pairs, internal, spin_rows, geo


def atom_label(i, z, u, rc):
    el = "C" if z[i] == 6 else "H"
    part = "C9" if i == rc else ("fl" if i in u["fluorenyl"] else ("an" if i in u["anthryl"] else "H"))
    return f"{el}{i}({part})"


# =========================
# PLOTS
# =========================
def projection_panel(ax_face, ax_side, geo, p, title, colors=(P_C, Q_C)):
    """Face view on the mean plane of the pair (frame of P: x = u_P long axis, y = v_P C2 axis) and an
    edge view (x = v_P, y = n_mean)."""
    pos, units, rc, rho = geo["pos"], geo["units"], geo["rc"], geo["rho"]
    a, b = PL[p]
    ia, ib = geo["sel"][a], geo["sel"][b]
    cP, nP, _ = fit_plane(pos[ia])
    cQ, nQ, _ = fit_plane(pos[ib])
    if nP @ nQ < 0:
        nQ = -nQ
    nm = unit_vec(nP + nQ)
    vP = unit_vec((pos[rc[a]] - cP) - ((pos[rc[a]] - cP) @ nm) * nm)
    uP = np.cross(nm, vP)
    origin = 0.5 * (cP + cQ)
    f2 = lambda x: np.column_stack([(x - origin) @ uP, (x - origin) @ vP])   # noqa: E731
    s2 = lambda x: np.column_stack([(x - origin) @ vP, (x - origin) @ nm])   # noqa: E731
    g = None
    smax = np.nanmax(np.abs(rho)) if rho is not None else 1
    for lab, col in ((a, colors[0]), (b, colors[1])):
        u = units[lab]
        for r in u[geo["ring_key"]]:
            ax_face.fill(*f2(pos[r]).T, color=col, alpha=0.22, lw=0)
            ax_face.plot(*np.vstack([f2(pos[r]), f2(pos[r])[:1]]).T, color=col, lw=1.2)
        other = "rings_an" if geo["ring_key"] == "rings_fl" else "rings_fl"
        for r in u[other]:
            ax_face.plot(*np.vstack([f2(pos[r]), f2(pos[r])[:1]]).T, color=col, lw=0.6, ls=":", alpha=0.7)
        c_ = pos[u["half_carbons"]]
        if rho is not None:
            s = rho[u["half_carbons"]]
            ax_face.scatter(*f2(c_).T, s=6 + 220 * np.abs(s) / smax, c=np.where(s > 0, INK, "white"),
                            edgecolor=INK, linewidth=0.5, zorder=4)
        ax_face.scatter(*f2(pos[[rc[lab]]]).T, marker="*", s=120, color=col, edgecolor=INK, lw=0.6, zorder=5)
        cen = pos[geo["sel"][lab]].mean(0)
        vec = unit_vec((pos[rc[lab]] - cen) - ((pos[rc[lab]] - cen) @ nm) * nm)
        c2 = f2(cen[None])[0]
        ax_face.annotate("", xy=c2 + 2.2 * np.array([vec @ uP, vec @ vP]), xytext=c2,
                         arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6), zorder=6)
        ax_face.text(*(c2 + [0.3, -0.6]), lab, color=col, fontsize=10, weight="bold", zorder=7)
        ax_side.scatter(*s2(pos[u["fluorenyl"]]).T, s=10, color=col)
        ax_side.scatter(*s2(pos[u["anthryl"]]).T, s=4, color=col, alpha=0.35)
        ax_side.scatter(*s2(pos[[rc[lab]]]).T, marker="*", s=90, color=col, edgecolor=INK, lw=0.5, zorder=5)
    ax_face.set_aspect("equal")
    ax_side.set_aspect("equal")
    ax_face.set_title(title, fontsize=8.5, color=INK)
    ax_face.set_xlabel("u (long axis of P), Å", fontsize=7)
    ax_face.set_ylabel("v (C2 axis of P → C9), Å", fontsize=7)
    ax_side.set_xlabel("v, Å", fontsize=7)
    ax_side.set_ylabel("n (mean normal), Å", fontsize=7)
    for ax in (ax_face, ax_side):
        style(ax)
        ax.tick_params(labelsize=6)
    return g


def make_plots(T, geos, sel_pairs, internal, plots):
    figs = {}
    # 1. projections of the selected pairs
    ncol = len(sel_pairs)
    fig, axes = plt.subplots(2, ncol, figsize=(3.3 * ncol, 6.6), gridspec_kw={"height_ratios": [3, 1.3]}, squeeze=False)
    for k, (key, p) in enumerate(sel_pairs):
        r = T[(T["key"] == key) & (T["pair"] == p)].iloc[0]
        ttl = (f"{key}  J_{p} ({PL[p]}) = {r['J_meV']:+.1f} meV\n"
               f"d⊥ {r['sep_normal_A']:.2f} Å  slip {r['slip_A']:.2f} Å  tilt {r['tilt_deg']:.0f}°\n"
               f"φ {r['phi_C2_deg']:.0f}°  overlap {r['overlap_area_A2']:.1f} Å²  |S| {r['S_mag']:.3f}")
        projection_panel(axes[0, k], axes[1, k], geos[key], p, ttl)
    fig.suptitle("Dominant intermolecular pairs: fluorenyl rings (filled) and anthryl rings (dotted) projected on the mean "
                 "plane;\nstar = C9, arrow = C2 axis (centroid → C9), dots = HS Mulliken spin (black > 0, white < 0, "
                 "size ∝ |ρ|)", fontsize=9)
    fig.tight_layout()
    figs["proj"] = "plots/pair_projections.png"
    fig.savefig(plots / "pair_projections.png", dpi=180)
    plt.close(fig)

    # 2. descriptors vs J (inter pairs)
    I = T[T["relation"] == "inter"]
    cols = [("sep_normal_A", "separation along mean normal (Å)"), ("slip_A", "lateral slip (Å)"),
            ("tilt_deg", "tilt between planes (°)"), ("phi_C2_deg", "in-plane rotation φ (°)"),
            ("overlap_area_A2", "projected ring overlap (Å²)"), ("contact_half_A", "closest contact (Å)"),
            ("mcconnell", "McConnell Σ ρaρb (C…C < 4 Å)"), ("S2_mag", "magnetic-orbital overlap S²")]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
    for ax, (c, lab) in zip(axes.ravel(), cols):
        for rel, sub in ((False, I[~I["reliable"]]), (True, I[I["reliable"]])):
            if rel:
                ax.scatter(sub[c], sub["J_meV"], c=np.where(sub["J_meV"] > 0, FM_C, AFM_C), s=np.where(sub["dominant"], 60, 22),
                           edgecolor=np.where(sub["dominant"], INK, "none"), lw=0.8, zorder=3)
            else:
                ax.scatter(sub[c], sub["J_meV"], facecolor="none", edgecolor=INK_3, s=16, lw=0.7, zorder=2)
        for _, r in I[I["reliable"] & I["dominant"] & (I["J_meV"].abs() > 4)].iterrows():
            ax.annotate(f"{r['key'].split('/')[1]}:{r['radicals']}", (r[c], r["J_meV"]), fontsize=6, color=INK_2,
                        xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color=INK_3, lw=0.8)
        if c == "S2_mag":
            ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel(lab, fontsize=8)
        ax.set_ylabel("J_ij (meV, >0 FM)", fontsize=8)
        style(ax)
    axes[0, 0].scatter([], [], color=FM_C, label="FM (reliable)")
    axes[0, 0].scatter([], [], color=AFM_C, label="AFM (reliable)")
    axes[0, 0].scatter([], [], facecolor="none", edgecolor=INK_3, label="unreliable 4-spin mapping")
    axes[0, 0].scatter([], [], color="white", edgecolor=INK, s=60, label="dominant inter pair")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Intermolecular J_ij versus fluorenyl–fluorenyl descriptors (all 4 inter pairs of every fitted dimer)", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    figs["desc"] = "plots/descriptors_vs_J.png"
    fig.savefig(plots / "descriptors_vs_J.png", dpi=180)
    plt.close(fig)

    # 3. slip map: slip components coloured by J
    R = I[I["reliable"] & (I["contact_half_A"] < 6)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    vmax = max(1, R["J_meV"].abs().max())
    from matplotlib.colors import SymLogNorm
    norm = SymLogNorm(linthresh=1, vmin=-vmax, vmax=vmax)
    sc = axes[0].scatter(R["slip_long_A"], R["slip_short_A"], c=R["J_meV"], cmap="RdBu_r", norm=norm,
                         s=40 + 400 * R["overlap_frac"], edgecolor=INK, lw=0.5)
    axes[0].set_xlabel("|slip| along long axis u_P (Å)")
    axes[0].set_ylabel("slip along C2 axis v_P (Å, > 0 towards C9 of P)")
    axes[0].set_title("Slip decomposition (marker size ∝ ring overlap fraction)", fontsize=9)
    axes[1].scatter(R["phi_C2_deg"], R["tilt_deg"], c=R["J_meV"], cmap="RdBu_r", norm=norm,
                    s=40 + 400 * R["overlap_frac"], edgecolor=INK, lw=0.5)
    axes[1].set_xlabel("in-plane rotation φ between C2 axes (°)")
    axes[1].set_ylabel("tilt between planes (°)")
    axes[1].set_title("Orientation (pairs with closest contact < 6 Å)", fontsize=9)
    for ax in axes:
        style(ax)
    for _, r in R[R["dominant"]].iterrows():
        axes[0].annotate(f"{r['key'].split('/')[1]}:{r['radicals']}", (r["slip_long_A"], r["slip_short_A"]), fontsize=6,
                         xytext=(4, 4), textcoords="offset points", color=INK_2)
        axes[1].annotate(f"{r['key'].split('/')[1]}:{r['radicals']}", (r["phi_C2_deg"], r["tilt_deg"]), fontsize=6,
                         xytext=(4, 4), textcoords="offset points", color=INK_2)
    fig.colorbar(sc, ax=axes, label="J_ij (meV, red FM / blue AFM)")
    figs["slip"] = "plots/slip_orientation_map.png"
    fig.savefig(plots / "slip_orientation_map.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 4. internal angles
    Iw = internal.copy()
    Iw = Iw.sort_values("key", key=lambda c: c.map(lambda k: tuple(natural_key(k))))
    keys = Iw["key"].unique().tolist()
    fig, ax = plt.subplots(figsize=(13, 4.6))
    for c, lab, col, m in (("theta_fl_an_deg", "fluorenyl–anthryl (outer torsion)", COLORS[0], "o"),
                           ("theta_an_an_deg", "anthryl–anthryl (central torsion)", COLORS[1], "s"),
                           ("fl_butterfly_deg", "fluorenyl butterfly (benzo–benzo)", COLORS[2], "^"),
                           ("an_bend_deg", "anthryl bend (outer–outer ring)", COLORS[3], "v")):
        x = Iw["key"].map(keys.index) + np.where(Iw["molecule"] == "M1", -0.15, 0.15)
        ax.scatter(x, Iw[c], color=col, marker=m, s=18, label=lab)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=90, fontsize=6.5)
    ax.set_ylabel("angle (°)")
    ax.set_title("Internal angles of each molecule (left of tick: molecule 1, right: molecule 2; two radicals each)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=4)
    style(ax)
    fig.tight_layout()
    figs["internal"] = "plots/internal_angles.png"
    fig.savefig(plots / "internal_angles.png", dpi=180)
    plt.close(fig)
    return figs


# =========================
# MAIN
# =========================
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plots = OUT / "plots"
    plots.mkdir(exist_ok=True)
    ref = Reference(REF_XYZ)
    J, E, F = load_inputs()
    pair_rows, int_rows, spin_rows, geos = [], [], [], {}
    for _, j in J.iterrows():
        key = f"{j['run']}/{j['dimer']}"
        e = E[(E["run"] == j["run"]) & (E["dimer"] == j["dimer"])]
        pairs, internal, spins, geo = analyse_dimer(j["run"], j["dimer"], ref, e)
        geos[key] = geo
        fam = F[F["key"] == key].iloc[0] if (F["key"] == key).any() else None
        fitted = j["fit"] != "insufficient"
        reliable = bool(fitted and fam is not None and fam["moments_ok"] and
                        not (np.isfinite(fam["max_rel_dJ"]) and fam["max_rel_dJ"] > RELIABLE_MAX_REL_DJ))
        jv = {p: j[f"J_{p}_meV"] for p in PAIRS}
        dom = max(INTER, key=lambda p: abs(jv[p]) if np.isfinite(jv[p]) else -1)
        for r in pairs:
            r.update({"key": key, "run": j["run"], "dimer": j["dimer"], "fit": j["fit"], "reliable": reliable,
                      "J_meV": jv[r["pair"]], "dominant": fitted and r["pair"] == dom,
                      "mag_family": fam["mag_family"] if fam is not None else "",
                      "struct_family": fam["struct_family"] if fam is not None else "",
                      "dE_geom_meV": fam["dE_geom_meV"] if fam is not None else np.nan})
        pair_rows += pairs
        int_rows += [{"key": key, **r} for r in internal]
        spin_rows += [{"key": key, **r} for r in spins]
        print(f"{key:16s} fit={j['fit']:14s} reliable={reliable!s:5s} S-eqs={pairs[0]['S_n_eq']} "
              f"dominant inter J_{dom}={jv[dom]:+.2f}")
    T = pd.DataFrame(pair_rows)
    first = ["key", "run", "dimer", "pair", "radicals", "relation", "J_meV", "dominant", "reliable", "fit",
             "mag_family", "struct_family", "dE_geom_meV"]
    T = T[first + [c for c in T.columns if c not in first]]
    T = T.sort_values(["key", "pair"], key=lambda c: c.map(lambda k: tuple(natural_key(k))) if c.name == "key" else c)
    T.to_csv(OUT / "fragment_pairs.csv", index=False)
    Iw = pd.DataFrame(int_rows)
    Iw.to_csv(OUT / "internal_angles.csv", index=False)
    S = pd.DataFrame(spin_rows)
    S.to_csv(OUT / "fragment_spins.csv", index=False)

    if SELECTED:
        sel = SELECTED
    else:
        D = T[T["dominant"] & T["reliable"]]
        sel = [(r["key"], r["pair"]) for _, r in D.nsmallest(N_SELECT_EACH, "J_meV").iterrows()] + \
              [(r["key"], r["pair"]) for _, r in D.nlargest(N_SELECT_EACH, "J_meV").iloc[::-1].iterrows()]
    figs = make_plots(T, geos, sel, Iw, plots)
    write_report(T, Iw, S, sel, figs, OUT / "overlap_analysis.md")
    print(f"wrote {OUT}")


# =========================
# REPORT
# =========================
def md_table(df, fmt):
    cols = list(fmt)
    L = ["| " + " | ".join(fmt[c][0] for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v, f = r[c], fmt[c][1]
            cells.append("" if (isinstance(v, float) and not np.isfinite(v)) else (f.format(v) if f else str(v)))
        L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L)


def write_report(T, Iw, S, sel, figs, path):
    from scipy.stats import spearmanr
    I = T[(T["relation"] == "inter") & T["reliable"]]
    nrel = I["key"].nunique()
    corr = []
    for c, lab in (("S2_mag", "S² (magnetic-orbital overlap)"), ("n_CC_4.0", "n C…C < 4 Å"),
                   ("d_C9_C9_A", "C9–C9 distance"), ("sep_normal_A", "separation along mean normal"),
                   ("c9_lateral_A", "C9 lateral offset"), ("tilt_deg", "tilt"), ("slip_A", "lateral slip"),
                   ("overlap_area_A2", "projected ring overlap"), ("phi_C2_deg", "in-plane rotation φ"),
                   ("mcconnell", "McConnell Σρaρb"), ("contact_half_A", "closest contact")):
        a, b = spearmanr(I[c], I["J_meV"]), spearmanr(I[c], I["J_meV"].abs())
        corr.append({"d": lab, "rJ": a.correlation, "pJ": a.pvalue, "rA": b.correlation, "pA": b.pvalue})
    corr = pd.DataFrame(corr)
    afm = I[I["J_meV"] < 0]
    slope = np.polyfit(afm["S2_mag"], afm["J_meV"], 1)[0]
    r_afm = spearmanr(afm["S2_mag"], afm["J_meV"]).correlation
    strong_afm, strong_fm = I[I["J_meV"] < -2], I[I["J_meV"] > 2]
    D = I[I["dominant"]].sort_values("J_meV")
    S1 = T[T["key"].isin(["run2/conf_7", "run2/conf_27", "run2/conf_28"]) & T["dominant"]]
    clos = T[(T["relation"] == "inter") & T["reliable"]]["S_closure"].max()

    m = Iw.groupby(["key", "molecule"]).agg(an_an=("theta_an_an_deg", "first"), fl_an=("theta_fl_an_deg", "mean"),
                                            bend=("an_bend_deg", "max"), butterfly=("fl_butterfly_deg", "max"),
                                            rms_an=("rms_anthryl_A", "max")).reset_index()
    intra = T[T["relation"] == "intra"][["key", "radicals", "J_meV", "dE_geom_meV", "reliable"]].copy()
    intra["molecule"] = intra["radicals"].map({"AB": "M1", "CD": "M2"})
    m = m.merge(intra, on=["key", "molecule"], how="left")
    m["class"] = np.where(m["an_an"] < 30, "planarised link", "twisted link")
    cls = m.groupby("class").agg(n=("key", "size"), an_an=("an_an", "median"), fl_an=("fl_an", "median"),
                                 bend=("bend", "median"), rms_an=("rms_an", "median"),
                                 J_min=("J_meV", "min"), J_max=("J_meV", "max"),
                                 dE_min=("dE_geom_meV", "min"), n_rel=("reliable", "sum")).reset_index()
    u = S[S["config"] == "uuuu"]
    u_rel = u[u["key"].isin(I["key"])]

    pair_fmt = {"key": ("dimer", None), "radicals": ("pair", None), "J_meV": ("J (meV)", "{:+.2f}"),
                "S2_mag": ("S²", "{:.4f}"), "S_mag": ("abs S", "{:.3f}"), "sep_normal_A": ("d⊥ (Å)", "{:.2f}"),
                "slip_A": ("slip (Å)", "{:.2f}"), "slip_short_A": ("slip v (Å)", "{:+.2f}"),
                "slip_long_A": ("slip u (Å)", "{:.2f}"), "tilt_deg": ("tilt (°)", "{:.0f}"),
                "phi_C2_deg": ("φ (°)", "{:.0f}"), "overlap_area_A2": ("overlap (Å²)", "{:.1f}"),
                "d_C9_C9_A": ("C9–C9 (Å)", "{:.2f}"), "contact_half_A": ("contact (Å)", "{:.2f}"),
                "contact_half_atoms": ("contact atoms", None), "n_CC_4.0": ("n C…C<4", "{:.0f}"),
                "mcconnell": ("McConnell", "{:+.3f}")}
    L = [
        "# Fragment overlap and magnetic-orbital overlap of the dimer radicals",
        "",
        "Script: `visualisation/analyze_dimer_overlap.py`. Data: [fragment_pairs.csv](fragment_pairs.csv) (one row per "
        "dimer × radical pair, all descriptors), [internal_angles.csv](internal_angles.csv), "
        "[fragment_spins.csv](fragment_spins.csv) (Mulliken spin of every unit, every finished BS configuration). "
        f"Reliable dimers = fitted, four stable moments and jackknife |ΔJ|/max|J| ≤ {RELIABLE_MAX_REL_DJ} "
        f"(as in [../families/dimer_families.md](../families/dimer_families.md)); {nrel} dimers, "
        f"{len(I)} intermolecular pairs. J convention: H = −Σ J S·S, J > 0 FM.",
        "",
        "## Main results",
        "",
        f"1. **The magnetic-orbital overlap S² is the only descriptor that follows the sign and size of the AFM "
        f"coupling.** Over the AFM pairs, J ≈ {slope:.0f} meV × S² (Spearman ρ = {r_afm:.2f}). "
        f"The strongest AFM contacts have S² = {strong_afm['S2_mag'].min():.4f}–{strong_afm['S2_mag'].max():.4f} "
        f"(|S| up to {strong_afm['S_mag'].max():.3f}). FM pairs with J > 2 meV have S² ≤ {strong_fm['S2_mag'].max():.4f}: "
        "their magnetic orbitals are nearly orthogonal. The FM and AFM ranges touch at about 0.0013–0.0014, "
        "so S² does not separate the sign perfectly on its own.",
        f"2. **Geometry sets the magnitude of J, not its sign.** The number of C…C contacts, the C9–C9 distance and "
        "the separation correlate with |J| but not with J (table below). Close contact is necessary for a sizable "
        "J of either sign; the sign is decided by the orbital overlap.",
        f"3. **The S01 FM motif (conf_7/27/28) is a near-eclipsed, parallel stack with nearly orthogonal magnetic "
        f"orbitals.** Tilt {S1['tilt_deg'].min():.0f}–{S1['tilt_deg'].max():.0f}°, φ {S1['phi_C2_deg'].min():.0f}–"
        f"{S1['phi_C2_deg'].max():.0f}° (C9 pointing the same way), separation {S1['sep_normal_A'].min():.2f}–"
        f"{S1['sep_normal_A'].max():.2f} Å and slip {S1['slip_A'].min():.1f}–{S1['slip_A'].max():.1f} Å, mostly "
        f"along the C2 axis. The projected ring overlap is the largest of all reliable pairs "
        f"({S1['overlap_area_A2'].min():.1f}–{S1['overlap_area_A2'].max():.1f} Å²), yet S² is only "
        f"{S1['S2_mag'].min():.4f}–{S1['S2_mag'].max():.4f}. This is the Kahn picture of FM coupling: the "
        "magnetic orbitals overlap in space but their positive and negative overlap regions cancel. A large π "
        "overlap area alone does not imply AFM coupling.",
        "4. **The strong AFM contacts are rotated or slipped stacks.** conf_11 AD (φ ≈ 48°), conf_6 AD (φ ≈ 35°) "
        "and run3/docked_4 BD (φ ≈ 161°, head-to-tail) all have separations of 3.3–3.8 Å, slips of 2.7–3.5 Å "
        "and a much larger S. Two strong FM pairs have no fluorenyl–fluorenyl ring overlap. conf_5 AC is "
        "head-to-tail (φ ≈ 165°) with a 5 Å slip along the C2 axis, which brings the two C9 to 3.98 Å. "
        "conf_33 BC (C9–C9 8.6 Å) and the AFM pair conf_14 BC (slip 10 Å) interact through anthryl contacts, "
        "which fluorenyl-plane descriptors cannot represent.",
        "5. **The HS McConnell-I sum does not predict the sign** (ρ ≈ 0 with J). S01 has the largest positive "
        "sum (same-sign spin contacts, McConnell predicts AFM) but is FM. Mulliken spin products on contact "
        "atoms are not a reliable sign descriptor here.",
        "6. **Internal geometry falls into two classes.** In every low-energy, reliable dimer the bianthryl link "
        "is twisted (θ_an–an ≈ 65–86°), the anthryls are nearly flat (bend ≤ 11°) and |J_intra| ≤ 5.2 meV "
        "(≤ 8 meV for all twisted-link molecules). In every dimer with a "
        "planarised link (θ_an–an < 30°), both anthracenes fold by about 40° (RMS 0.4 Å). These dimers have "
        "J_intra ≈ −330 to −530 meV, lie > 1.7 eV above the minimum and all fail the four-spin mapping.",
        "",
        "## Descriptor ranking (reliable intermolecular pairs)",
        "",
        "| descriptor | ρ(J) | p | ρ(abs J) | p |",
        "|---|---|---|---|---|",
    ]
    for _, r in corr.iterrows():
        L.append(f"| {r['d']} | {r['rJ']:+.2f} | {r['pJ']:.2g} | {r['rA']:+.2f} | {r['pA']:.2g} |")
    L += ["", "Spearman rank correlations, so the numbers are insensitive to outliers and units.", "",
          "## Dominant intermolecular pair of every reliable dimer", "",
          md_table(D, pair_fmt), "",
          "slip v > 0: centroid of the second fragment displaced towards the C9 of the first. φ = 0: both C2 axes "
          "point the same way; 180°: head-to-tail. contact atoms: index (0-based, ORCA) and unit (fl = fluorenyl, "
          "an = anthryl, C9 = radical carbon, H).", "",
          f"![projections]({figs['proj']})", "",
          "Face view on the mean plane (frame of the first fragment) and edge view, for the three strongest AFM and "
          "three strongest FM dominant pairs" + ("" if SELECTED else " (automatic selection)") + ".", "",
          f"![descriptors]({figs['desc']})", "",
          f"![slip]({figs['slip']})", "",
          "## Definitions", "",
          "**Fragments.** Each radical half (A, B on molecule 1; C, D on molecule 2) is read from "
          "`radical_fragments.json`, the same atoms used for FlipSpin and the J fit. It consists of a 13-C "
          "fluorenyl (5-ring with the radical carbon C9 plus two benzo rings) bonded through C9 to a 14-C "
          "anthracene-type unit (three linearly fused rings). The two anthryls are joined by the central bond. "
          f"The units are defined in the reference numbering (`REF_UNITS`) and mapped by graph isomorphism. "
          f"Pair descriptors use `PLANE_UNIT = \"{PLANE_UNIT}\"`; `half_carbons`, `anthryl` or `custom` "
          "(`CUSTOM_REF_UNITS`) are alternatives. The fluorenyl is the natural radical fragment: in uuuu it carries "
          f"{u_rel['frac_fluorenyl'].median():.0%} of the half's spin (range {u_rel['frac_fluorenyl'].min():.0%}–"
          f"{u_rel['frac_fluorenyl'].max():.0%}), with {u_rel['spin_C9'].median():.2f} on C9 alone.", "",
          "**Planes and distances.** Each plane is an SVD best-plane fit, and planarity is the RMS out-of-plane "
          "distance of the unit (fluorenyl ≤ 0.1 Å everywhere). The plane normals are aligned and averaged into "
          "n̄. Separation d⊥ is the centroid difference projected on n̄. Slip is the remaining in-plane part. "
          "Tilt is the angle between the normals.", "",
          "**In-plane rotation.** Parallel planes do not fix the overlap: two parallel fluorenyls can still be "
          "rotated or slipped, which changes the SOMO overlap and can flip the sign of J. The in-plane frame is "
          "built from atom coordinates only. v is the projection of (C9 − fluorenyl centroid) on the mean plane, "
          "which is the C2 axis of fluorenyl and a polar direction. u = n̄ × v is the long axis; its sign is "
          "undefined because the two benzo rings are equivalent, so only |slip_u| is reported. φ is the angle "
          "between v_P and v_Q (0–180°), and χ_long = min(φ, 180° − φ) is the long-axis crossing angle. Above "
          f"about {TILT_WARN:.0f}° tilt, φ and the slip split are only indicative.", "",
          "**Overlap area.** The three ring polygons of each fluorenyl are projected on the mean plane and the area "
          f"of their intersection is rasterised (step {GRID_STEP} Å). This is a geometric π-stacking measure "
          "that does not use orbital phases.", "",
          f"**McConnell-I sum.** Σ ρa ρb over the C…C pairs closer than {MC_CUT} Å between the two halves, with "
          "HS (uuuu) Mulliken spin populations. In McConnell's model, a positive sum (contacts of same-sign spin "
          "density) favours AFM coupling and a negative sum favours FM coupling.", "",
          "## Spin and density information: what the ORCA outputs provide", "",
          "Available per BS configuration: `.out` (Mulliken and Löwdin atomic spin populations, UNO occupation "
          "numbers, UCO overlaps), `.gbw`, `.uno`, `.uco`, `.qro`, `.property.txt` and `.densitiesinfo`. Most "
          "`.densities` files have been deleted. There are no `.cube` files for the dimers (the 36 cubes in the "
          "repository belong to the monomer `angle_scans/natural_orbitals` analysis).", "",
          "1. **Integrated spin on the fragments (implemented).** Mulliken spins are summed over fluorenyl, anthryl "
          "and H of every half for every finished configuration ([fragment_spins.csv](fragment_spins.csv)). "
          "This is reliable and cheap, but it only measures how much spin each unit carries, not how strongly two "
          "units interact.",
          "2. **Magnetic-orbital overlap S_ij (implemented; the main density-based descriptor).** With UCO on, "
          "ORCA prints the overlaps T of the corresponding α/β orbital pairs. In a BS state with k flipped radicals, "
          "the k smallest T are the overlaps between the β magnetic orbitals of the flipped radicals and the α "
          "magnetic orbitals of the others. The list has N_β + 1 entries; the last one is an unmatched α orbital "
          "printed as 0. A fixed cutoff would fail because polarised core pairs can reach 0.87 when the coupling is "
          "strong. With x_ij = S_ij²: a single flip of i gives T_i² = Σ_j x_ij, and a pair flip (ij | kl) gives "
          "T₁² + T₂² = x_ik + x_il + x_jk + x_jl. The 7 BS states therefore give 7 linear equations for the "
          "6 S_ij², with one redundant closure check. The largest closure residual among the reliable dimers is "
          f"{clos:.1e}, against S² ≈ 10⁻³. Physical meaning: S_ij = ⟨a_i|b_j⟩ is the overlap between the "
          "magnetic orbitals of radicals i and j. The integrand a_i(r) b_j(r) vanishes except where both orbitals "
          "are non-zero, i.e. in the intermolecular contact region, so S_ij is exactly the contact overlap of the "
          "SOMOs. In the Kahn / Hay–Thibeault–Hoffmann picture, J = J_F + J_AF with J_AF ∝ −S² (kinetic, "
          "antiferromagnetic) and J_F > 0 from the exchange integral of the overlap density. Limitations: only |S| "
          "is recovered (the sign is lost), the magnetic orbitals are assumed transferable between configurations "
          "(the same assumption as the Ising fit), and only configurations with `used_in_fit` enter.",
          "3. **Cube-based measures (feasible, not implemented).** `orca_plot` 6.1.1 runs locally. For a BS "
          "`.gbw` it lists the SCF electron and spin densities as available (stored in the `.gbw`), and it can "
          "plot any MO, UNO or UCO. A 40³ spin-density cube of one dimer takes about 40 s. The physically "
          "motivated next step would target J_F, the ferromagnetic part that S² misses and that matters for "
          "S01: plot the two magnetic corresponding orbitals of a pair (cleanest from the single-flip states, "
          "where the β magnetic orbital is localised on one radical) and integrate the overlap density "
          "ρ_ab = a·b. ∫ρ_ab reproduces S, as a check; ∫ρ_ab² (or ∫|ρ_ab|) measures how much the orbitals "
          "coexist in space and is the usual proxy for the exchange integral K_ab, which drives FM coupling when "
          "S ≈ 0. A large ∫ρ_ab² with small |S| would confirm the accidental-orthogonality explanation of S01. "
          "A simple spin-density overlap (∫ρ_s,A ρ_s,B) is not proposed: it has no direct link to J and it "
          "repeats the McConnell sum, which already fails here.", "",
          "## Internal angles of the molecules", "",
          "Per radical half: θ_fl–an = angle between the fluorenyl and anthryl planes (outer torsion); fluorenyl "
          "butterfly = angle between its two benzo rings; anthryl bend = angle between its two outer rings. Per "
          "molecule: θ_an–an (central torsion) and θ_fl–fl (end-to-end). These are whole-unit SVD planes, so the "
          "values differ by a few degrees from the ring-to-ring angles in the families report. In run3 the A/C "
          "outer torsion is about 88° because those relaxations had constrained torsions.", "",
          md_table(cls, {"class": ("class", None), "n": ("molecules", "{:.0f}"), "n_rel": ("in reliable dimers", "{:.0f}"),
                         "an_an": ("median θ_an–an (°)", "{:.0f}"), "fl_an": ("median θ_fl–an (°)", "{:.0f}"),
                         "bend": ("median anthryl bend (°)", "{:.0f}"), "rms_an": ("median anthryl RMS (Å)", "{:.2f}"),
                         "J_min": ("J_intra min (meV)", "{:+.0f}"), "J_max": ("J_intra max (meV)", "{:+.1f}"),
                         "dE_min": ("lowest ΔE_geom (meV)", "{:.0f}")}), "",
          f"![internal]({figs['internal']})", "",
          "## Caveats", "",
          "- 14 reliable dimers with 56 intermolecular pairs, and only 5–7 strong pairs of each sign: the trends are "
          "robust in rank but thresholds are not.",
          "- S² relies on the ORCA UCO list and on transferable magnetic orbitals; for the unreliable dimers "
          "(pattern-failed configurations removed) it is often underdetermined or noisy.",
          "- Mulliken spins (fragment spins, McConnell) depend on the basis; Löwdin spins are in the same outputs if "
          "a cross-check is needed.",
          "- Fluorenyl-plane descriptors miss couplings through anthryl contacts (conf_5 AC, conf_33 BC); set "
          "`PLANE_UNIT = \"half_carbons\"` to describe the whole halves.",
          ""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
