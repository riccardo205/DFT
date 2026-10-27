#!/usr/bin/env python3
"""
Analysis of the full ORCA constrained-dihedral (torsion) high-spin geometry
scan: all L x C x R combinations of a 1-2-3-4 four-fragment molecule
(radicals on terminal fragments 1 and 4).

Organized around 7 scientific questions (see angle_scan.md section headers):
  1. accessible L,C,R space (convergence + chemical integrity)
  2. central-torsion accommodation (plane angle, bond tilt, puckering)
  3. 45/45 <-> 135/135 outer-folding crossover across C~90
  4. whether 1-3 / 2-4 cross-fragment contacts drive that crossover
  5. independence of outer torsions L, R in controlling terminal spin
  6. correlation between terminal spin localization and AFM HS-BS exchange
     (scaffold only until the separate PBE0 HS/BS single points exist)
  7. any geometry where the triplet becomes competitive with/lower than BS

Exchange quantities: DeltaE = E_HS - E_BS (column delta_E_exchange_meV; > 0: BS lower). J in the
repository convention H = -J S1.S2 (J > 0 FM) is computed in analyze_exchange.py via exchange_convention.py.
"""

import re
import warnings
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read
from ase.neighborlist import NeighborList, get_connectivity_matrix, natural_cutoffs

# ============================================================
# REGEX PATTERNS
# ============================================================
MULLIKEN_BLOCK_RE = re.compile(
    r"MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS\s*\n-+\n(.*?)\n"
    r"Sum of atomic spin populations:\s*-?\d+\.\d+",
    re.S,
)
ATOM_LINE_RE = re.compile(r"\s*(\d+)\s+(\S+)\s*:\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)")
S2_RE = re.compile(r"Expectation value of <S\*\*2>\s*:\s*(-?\d+\.\d+)")
FINAL_E_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
DISPERSION_RE = re.compile(r"Dispersion correction\s+(-?\d+\.\d+)\s*$", re.MULTILINE)
CYCLE_RE = re.compile(r"GEOMETRY OPTIMIZATION CYCLE\s*(\d+)")
SCF_CYCLES_RE = re.compile(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES")
SCF_NOT_CONVERGED_RE = re.compile(r"SCF NOT CONVERGED")
TRAH_RE = re.compile(r"\(TRAH")
NEG_HESSIAN_RE = re.compile(r"\d+ diagonal Hessian elements are negative")
CONSTRAINT_RE = re.compile(r"\{\s*D\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+\.\d+)\s+C\s*\}")
CONV_TABLE_RE = re.compile(
    r"Energy change\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(YES|NO)\s+"
    r"RMS gradient\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(YES|NO)\s+"
    r"MAX gradient\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(YES|NO)\s+"
    r"RMS step\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(YES|NO)\s+"
    r"MAX step\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(YES|NO)"
)
CONV_CRITERIA_NAMES = ["energy_change", "rms_gradient", "max_gradient", "rms_step", "max_step"]
FOLDER_NAME_RE = re.compile(r"^L(\d+)_C(\d+)_R(\d+)$")
CONVERGED_BANNER = "THE OPTIMIZATION HAS CONVERGED"

HARTREE_TO_MEV = 27211.386245988


# ============================================================
# CONNECTIVITY / RING / FRAGMENT HELPERS
# ============================================================
def build_adjacency(atoms):
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    conn = get_connectivity_matrix(nl.nl, sparse=True).toarray()
    n = conn.shape[0]
    return {i: [j for j in range(n) if conn[i, j]] for i in range(n)}


def _shortest_path(adj, start, end, forbidden_edges):
    forb = set()
    for a, b in forbidden_edges:
        forb.add((a, b))
        forb.add((b, a))
    prev = {start: None}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        if u == end:
            break
        for v in adj[u]:
            if (u, v) in forb or v in prev:
                continue
            prev[v] = u
            queue.append(v)
    if end not in prev:
        return None
    path = [end]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1]


def smallest_ring_at(adj, pivot, exclude_neighbor):
    others = [x for x in adj[pivot] if x != exclude_neighbor]
    best = None
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            a, b = others[i], others[j]
            path = _shortest_path(adj, a, b, forbidden_edges=[(pivot, a), (pivot, b)])
            if path is None:
                continue
            ring = [pivot] + path
            if best is None or len(ring) < len(best):
                best = ring
    return best


def _reachable_avoiding_edge(adj, start, avoid_edge):
    b, c = avoid_edge
    visited = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if (u, v) in ((b, c), (c, b)):
                continue
            if v not in visited:
                visited.add(v)
                stack.append(v)
    return visited


def outer_terminal_fragment(adj, bond, core_markers):
    b, c = bond
    comp_b = _reachable_avoiding_edge(adj, b, bond)
    comp_c = _reachable_avoiding_edge(adj, c, bond)
    core = set(core_markers)
    if not (comp_b & core):
        return sorted(comp_b)
    if not (comp_c & core):
        return sorted(comp_c)
    raise RuntimeError(f"Could not identify outer terminal fragment for bond {bond}: both sides touch core markers")


def split_fragment_at_bond(adj, bond, fragment_atoms):
    """Split `fragment_atoms` into the two halves separated by `bond`=(b,c)."""
    b, c = bond
    comp_b = _reachable_avoiding_edge(adj, b, bond) & set(fragment_atoms)
    comp_c = set(fragment_atoms) - comp_b
    return sorted(comp_b), sorted(comp_c)


# ============================================================
# ORCA .inp / .out PARSING
# ============================================================
def parse_constraints(inp_text):
    return [
        (int(a), int(b), int(c), int(d), float(val))
        for a, b, c, d, val in CONSTRAINT_RE.findall(inp_text)
    ]


def match_quartet(constraints, quartet):
    a, b, c, d = quartet
    for ca, cb, cc, cd, val in constraints:
        if (ca, cb, cc, cd) in ((a, b, c, d), (d, c, b, a)):
            return val
    return None


def discover_all_scan_points(root, center, left, right):
    points = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        inp_file = folder / "high_spin.inp"
        if not inp_file.exists():
            continue
        constraints = parse_constraints(inp_file.read_text(errors="ignore"))
        c_val = match_quartet(constraints, center)
        l_val = match_quartet(constraints, left)
        r_val = match_quartet(constraints, right)
        if c_val is None or l_val is None or r_val is None:
            warnings.warn(f"{folder.name}: could not find all 3 constraints, skipping")
            continue
        points.append({"folder": folder, "L": l_val, "C": c_val, "R": r_val})
    return points


def classify_convergence(out_text):
    """
    Authoritative convergence = the literal ORCA banner is present. The type
    of override (if any) is then read off the message ORCA prints just above
    the banner, purely for reporting -- it does not affect the converged/not
    decision, per instruction not to reconstruct convergence from the YES/NO
    table alone.
    """
    idx = out_text.rfind(CONVERGED_BANNER)
    if idx == -1:
        return False, "not_converged", {}

    window = out_text[max(0, idx - 3000) : idx]
    conv_idx = out_text.rfind("|Geometry convergence|", 0, idx)
    criteria = {}
    if conv_idx != -1:
        m = CONV_TABLE_RE.search(out_text, conv_idx)
        if m:
            g = m.groups()
            for i, name in enumerate(CONV_CRITERIA_NAMES):
                val, tol, flag = g[3 * i], g[3 * i + 1], g[3 * i + 2]
                criteria[name] = {"value": float(val), "tolerance": float(tol), "converged": flag == "YES"}

    if "gradient convergence is overachieved" in window:
        conv_type = "gradient_overachieved"
    elif "step convergence is overachieved" in window:
        conv_type = "step_overachieved"
    elif "Everything but the energy has converged" in window:
        conv_type = "energy_override"
    elif criteria and all(c["converged"] for c in criteria.values()):
        conv_type = "strict"
    else:
        conv_type = "other_override"

    return True, conv_type, criteria


def parse_out_file(out_text, out_path):
    rec = {}
    rec["terminated_normally"] = "ORCA TERMINATED NORMALLY" in out_text

    energies = [float(x) for x in FINAL_E_RE.findall(out_text)]
    rec["final_energy"] = energies[-1] if energies else None

    dispersions = [float(x) for x in DISPERSION_RE.findall(out_text)]
    rec["final_dispersion_correction"] = dispersions[-1] if dispersions else None

    s2_values = [float(x) for x in S2_RE.findall(out_text)]
    rec["s2_initial"] = s2_values[0] if s2_values else None

    rec["n_geom_cycles"] = len(CYCLE_RE.findall(out_text))

    scf_cycles = [int(x) for x in SCF_CYCLES_RE.findall(out_text)]
    rec["scf_cycles_mean"] = float(np.mean(scf_cycles)) if scf_cycles else None
    rec["scf_cycles_max"] = max(scf_cycles) if scf_cycles else None

    rec["scf_not_converged"] = bool(SCF_NOT_CONVERGED_RE.search(out_text))
    rec["trah_activations"] = len(TRAH_RE.findall(out_text))
    rec["neg_hessian_events"] = len(NEG_HESSIAN_RE.findall(out_text))

    slurm_err = out_path.parent / "slurm.err"
    rec["time_limit_cancelled"] = (
        slurm_err.exists() and "DUE TO TIME LIMIT" in slurm_err.read_text(errors="ignore")
    )

    converged, conv_type, criteria = classify_convergence(out_text)
    rec["opt_converged"] = converged
    rec["convergence_type"] = conv_type
    rec["conv_criteria"] = criteria

    # A "final" spin population requires the optimization to have actually
    # converged AND ORCA to have printed a post-optimization population
    # analysis (i.e. more than just the initial-SCF block). Anything else
    # only gets a "last_available_spin" for bookkeeping, kept out of all
    # primary plots and never treated as a relaxed spin state.
    blocks = list(MULLIKEN_BLOCK_RE.finditer(out_text))

    def parse_block(block_text):
        spins = {}
        for line in block_text.splitlines():
            m = ATOM_LINE_RE.match(line)
            if m:
                spins[int(m.group(1))] = float(m.group(4))
        return spins

    rec["spin_initial"] = parse_block(blocks[0].group(1)) if blocks else {}
    rec["last_available_spin"] = parse_block(blocks[-1].group(1)) if blocks else {}
    has_final_analysis = len(blocks) >= 2
    rec["spin_final"] = parse_block(blocks[-1].group(1)) if (converged and has_final_analysis) else {}
    rec["s2_final"] = (s2_values[-1] if s2_values else None) if (converged and has_final_analysis) else None
    rec["has_final_spin"] = converged and has_final_analysis

    return rec


def classify_status(rec):
    if rec["opt_converged"]:
        return "converged"
    if rec["final_energy"] is None:
        return "failed (walltime)" if rec["time_limit_cancelled"] else "failed"
    if not rec["terminated_normally"]:
        return "incomplete"
    return "not converged"


# ============================================================
# CHEMICAL INTEGRITY CHECK
# ============================================================
def reference_bonds(ref_adj, ref_atoms):
    pos = ref_atoms.get_positions()
    bonds = []
    n = len(ref_atoms)
    for i in range(n):
        for j in ref_adj[i]:
            if j > i:
                bonds.append((i, j, float(np.linalg.norm(pos[i] - pos[j]))))
    return bonds


def check_chemical_integrity(atoms, bonds, bonded_pairs, bond_delta_tol, bond_abs_max, min_nonbonded_dist):
    pos = atoms.get_positions()
    reasons = []
    broken_bonds = []

    for i, j, ref_len in bonds:
        d = float(np.linalg.norm(pos[i] - pos[j]))
        stretch = d - ref_len
        if stretch > bond_delta_tol or d > bond_abs_max:
            broken_bonds.append((i, j, ref_len, d))
            reasons.append(f"bond {i}-{j} stretched {ref_len:.3f}→{d:.3f} Å")

    dmat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    for i, j in bonded_pairs:
        dmat[i, j] = np.inf
        dmat[j, i] = np.inf
    min_dist = float(dmat.min())
    i, j = (int(x) for x in np.unravel_index(np.argmin(dmat), dmat.shape))

    clash_pairs = {
        (int(a), int(b)) for a, b in zip(*np.where(np.triu(dmat < min_nonbonded_dist, k=1)))
    }
    for a, b in sorted(clash_pairs):
        reasons.append(f"non-bonded clash: atoms {a}-{b} at {dmat[a, b]:.3f} Å")

    broken_bond_pairs = {(bi, bj) for bi, bj, _, _ in broken_bonds}
    return {
        "valid": (len(broken_bond_pairs) == 0) and (len(clash_pairs) == 0),
        "bond_broken": len(broken_bond_pairs) > 0,
        "clash": len(clash_pairs) > 0,
        "broken_bond_pairs": broken_bond_pairs,
        "clash_pairs": clash_pairs,
        "reasons": reasons,
        "min_nonbonded_distance": min_dist,
        "min_nonbonded_pair": (i, j),
    }


# ============================================================
# STRUCTURAL ANALYSIS
# ============================================================
def fit_plane(coords):
    centroid = coords.mean(axis=0)
    _, _, vt = np.linalg.svd(coords - centroid)
    normal = vt[-1]
    return centroid, normal


def plane_rms_max_dev(coords, centroid, normal):
    d = (coords - centroid) @ normal
    return float(np.sqrt(np.mean(d**2))), float(np.max(np.abs(d)))


def oriented_normal(normal, own_centroid, other_centroid):
    """Flip `normal` if needed so it points from own_centroid toward other_centroid."""
    return normal if np.dot(normal, other_centroid - own_centroid) >= 0 else -normal


def pivot_bending(pos, pivot_idx, ring_atoms_excl_pivot, own_centroid, other_centroid):
    """
    Signed out-of-plane displacement of the pivot atom from the plane fit to
    the REST of its own ring (excluding the pivot). POSITIVE = pivot displaced
    toward the other aromatic unit (inward), NEGATIVE = away (outward).
    """
    centroid, normal = fit_plane(pos[ring_atoms_excl_pivot])
    normal = oriented_normal(normal, own_centroid, other_centroid)
    return float(np.dot(pos[pivot_idx] - centroid, normal))


def bond_tilt(pos, atom_from, atom_to, plane_normal_oriented):
    """
    Signed out-of-plane tilt of the bond atom_from->atom_to relative to the
    (already outward/inward-oriented) aromatic-plane normal. POSITIVE = the
    bond tilts toward the neighboring/opposite fragment, NEGATIVE = away.
    """
    vec = pos[atom_to] - pos[atom_from]
    unit = vec / np.linalg.norm(vec)
    return float(np.degrees(np.arcsin(np.clip(np.dot(unit, plane_normal_oriented), -1.0, 1.0))))


def analyze_geometry(xyz_path, dihedral, left_ring, right_ring):
    atoms = read(xyz_path)
    a, b, c, d = dihedral

    result = {
        "dihedral_actual": atoms.get_dihedral(a, b, c, d),
        "bond_ab": atoms.get_distance(a, b),
        "bond_bc": atoms.get_distance(b, c),
        "bond_cd": atoms.get_distance(c, d),
        "angle_abc": atoms.get_angle(a, b, c),
        "angle_bcd": atoms.get_angle(b, c, d),
    }

    pos = atoms.get_positions()
    left_centroid, left_normal_raw = fit_plane(pos[left_ring])
    right_centroid, right_normal_raw = fit_plane(pos[right_ring])
    left_normal = oriented_normal(left_normal_raw, left_centroid, right_centroid)
    right_normal = oriented_normal(right_normal_raw, right_centroid, left_centroid)

    left_rms, left_max = plane_rms_max_dev(pos[left_ring], left_centroid, left_normal)
    right_rms, right_max = plane_rms_max_dev(pos[right_ring], right_centroid, right_normal)
    result["left_plane_rms"] = left_rms
    result["left_plane_max"] = left_max
    result["right_plane_rms"] = right_rms
    result["right_plane_max"] = right_max

    cos_angle = np.clip(abs(np.dot(left_normal, right_normal)), -1.0, 1.0)
    result["central_plane_angle"] = float(np.degrees(np.arccos(cos_angle)))

    left_ring_excl_b = [i for i in left_ring if i != b]
    right_ring_excl_c = [i for i in right_ring if i != c]
    result["bend_b"] = pivot_bending(pos, b, left_ring_excl_b, left_centroid, right_centroid)
    result["bend_c"] = pivot_bending(pos, c, right_ring_excl_c, right_centroid, left_centroid)

    result["tilt_ab"] = bond_tilt(pos, a, b, left_normal)
    result["tilt_cd"] = bond_tilt(pos, d, c, right_normal)

    return result, atoms


def fold_to_0_90(angle_deg):
    a = angle_deg % 180.0
    return a if a <= 90.0 else 180.0 - a


def magnitude(theta):
    return min(theta, 180.0 - theta)


def handedness(theta):
    return float(np.sign(theta - 90.0))


def angle_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def cross_fragment_min_distance(pos, symbols, frag_a, frag_b, heavy_only):
    idx_a = [i for i in frag_a if (not heavy_only or symbols[i] != "H")]
    idx_b = [i for i in frag_b if (not heavy_only or symbols[i] != "H")]
    sub = np.linalg.norm(pos[idx_a][:, None, :] - pos[idx_b][None, :, :], axis=-1)
    flat_argmin = np.argmin(sub)
    ia, ib = np.unravel_index(flat_argmin, sub.shape)
    return float(sub[ia, ib]), (idx_a[ia], idx_b[ib])


def cross_fragment_contact_stats(pos, symbols, frag_a, frag_b, n_shortest, cutoff):
    """
    Robust heavy-atom packing descriptors between two fragments, less fragile
    than a single minimum distance: mean of the `n_shortest` shortest
    heavy-atom pair distances, and the number of pairs closer than `cutoff`.
    """
    idx_a = [i for i in frag_a if symbols[i] != "H"]
    idx_b = [i for i in frag_b if symbols[i] != "H"]
    dists = np.sort(np.linalg.norm(pos[idx_a][:, None, :] - pos[idx_b][None, :, :], axis=-1).ravel())
    return float(dists[:n_shortest].mean()), int((dists < cutoff).sum())


def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, int(mask.sum())
    return float(np.corrcoef(x[mask], y[mask])[0, 1]), int(mask.sum())


# ============================================================
# MAIN
# ============================================================
def main():
    # ---- USER SETTINGS ----
    REPO_ROOT = Path(__file__).resolve().parent.parent
    ROOT_DIR = REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans" / "angle_scan_relaxation"
    OUTPUT_DIR = ROOT_DIR

    CENTER = (4, 9, 17, 18)
    LEFT = (42, 41, 0, 5)
    RIGHT = (21, 23, 28, 29)
    RADICAL_ATOMS = (28, 41)  # supplementary only; fragment-integrated spin is primary

    ANGLE_TOLERANCE_DEG = 1.0
    IDEAL_S2_TRIPLET = 2.0
    S2_ANOMALY_THRESHOLD = 0.30
    PLANARITY_RMS_THRESHOLD_ANG = 0.05

    BOND_STRETCH_TOL_ANG = 0.30
    BOND_ABS_MAX_ANG = 1.9
    MIN_NONBONDED_DIST_ANG = 1.5

    # Robust 1-3 / 2-4 packing descriptors
    N_SHORTEST_CONTACTS = 3
    CONTACT_CUTOFF_ANG = 4.0

    # A jump in energy + spin between grid neighbors is NOT called an
    # electronic/SCF branch change if the two structures also differ strongly
    # in these structural-accommodation descriptors (then the change has a
    # structural explanation).
    STRUCTURAL_JUMP_PLANE_ANGLE_DEG = 8.0
    STRUCTURAL_JUMP_PLANARITY_RMS_ANG = 0.02
    STRUCTURAL_JUMP_TILT_DEG = 3.0

    ENERGY_DISCONTINUITY_FACTOR = 2.5
    ENERGY_DISCONTINUITY_FLOOR_MEV = 20.0
    S2_DISCONTINUITY_FLOOR = 0.02
    SPIN_DISCONTINUITY_FLOOR = 0.03

    PLOTS_DIR = OUTPUT_DIR / "plots"
    HEATMAPS_DIR = PLOTS_DIR / "heatmaps"
    MD_PATH = OUTPUT_DIR / "angle_scan.md"
    CSV_PATH = OUTPUT_DIR / "angle_scan.csv"

    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"Scan root not found: {ROOT_DIR.resolve()}")

    points = discover_all_scan_points(ROOT_DIR, CENTER, LEFT, RIGHT)
    if not points:
        raise RuntimeError(f"No scan points found under {ROOT_DIR}")
    print(f"Found {len(points)} scan points (full L x C x R grid).")

    # ---- REFERENCE GEOMETRY / CONNECTIVITY (clean, untwisted baseline) ----
    ref_geom_path = REPO_ROOT / "scripts" / "orca" / "orca_functional_scan_setup" / "geometries" / "relaxed_main_conf_4.xyz"
    ref_atoms = read(ref_geom_path)
    adj = build_adjacency(ref_atoms)
    n_atoms = len(ref_atoms)
    bonded_pairs = {(i, j) for i in range(n_atoms) for j in adj[i] if j > i}
    bonds = reference_bonds(adj, ref_atoms)

    a, b, c, d = CENTER
    left_ring = smallest_ring_at(adj, b, c)
    right_ring = smallest_ring_at(adj, c, b)
    print(f"LEFT_PLANE_ATOMS (ring at atom {b}): {left_ring}")
    print(f"RIGHT_PLANE_ATOMS (ring at atom {c}): {right_ring}")

    core_markers = set(CENTER)
    fragment1 = outer_terminal_fragment(adj, (LEFT[1], LEFT[2]), core_markers)   # terminal, radical atom 41
    fragment4 = outer_terminal_fragment(adj, (RIGHT[1], RIGHT[2]), core_markers)  # terminal, radical atom 28
    center_fragment = sorted(set(range(n_atoms)) - set(fragment1) - set(fragment4))
    fragment2, fragment3 = split_fragment_at_bond(adj, (b, c), center_fragment)  # split center at the CENTER bond
    print(f"Fragment 1 (LEFT terminal,  {len(fragment1)} atoms): {fragment1}")
    print(f"Fragment 2 (backbone, B-side, {len(fragment2)} atoms)")
    print(f"Fragment 3 (backbone, C-side, {len(fragment3)} atoms)")
    print(f"Fragment 4 (RIGHT terminal, {len(fragment4)} atoms): {fragment4}")
    print()

    # ---- PARSE EVERY SCAN POINT ----
    rows = []
    for pt in points:
        folder = pt["folder"]
        label = folder.name
        out_file = folder / "high_spin.out"
        geometry_file = folder / "geometry.xyz"
        final_xyz = folder / "high_spin.xyz"

        row = {"label": label, "L": pt["L"], "C": pt["C"], "R": pt["R"]}
        for axis in ("L", "C", "R"):
            row[f"{axis}_magnitude"] = magnitude(row[axis])
            row[f"{axis}_handedness"] = handedness(row[axis])

        name_match = FOLDER_NAME_RE.match(label)
        if name_match:
            l_name, c_name, r_name = (float(x) for x in name_match.groups())
            if abs(l_name - pt["L"]) > ANGLE_TOLERANCE_DEG or abs(c_name - pt["C"]) > ANGLE_TOLERANCE_DEG or abs(r_name - pt["R"]) > ANGLE_TOLERANCE_DEG:
                warnings.warn(f"{label}: folder name disagrees with .inp constraints")

        if not out_file.exists():
            row["status"] = "no .out found"
            rows.append(row)
            warnings.warn(f"{label}: no high_spin.out found")
            continue

        out_text = out_file.read_text(errors="ignore")
        rec = parse_out_file(out_text, out_file)
        row.update(rec)
        row["status"] = classify_status(rec)

        row["radical_spin_initial"] = {atom: rec["spin_initial"].get(atom) for atom in RADICAL_ATOMS}
        row["radical_spin_final"] = {atom: rec["spin_final"].get(atom) for atom in RADICAL_ATOMS}
        row["radical_spin_last_available"] = {atom: rec["last_available_spin"].get(atom) for atom in RADICAL_ATOMS}

        if rec["has_final_spin"]:
            row["spin_fragment1_final"] = sum(rec["spin_final"].get(i, 0.0) for i in fragment1)
            row["spin_fragment2_final"] = sum(rec["spin_final"].get(i, 0.0) for i in fragment2)
            row["spin_fragment3_final"] = sum(rec["spin_final"].get(i, 0.0) for i in fragment3)
            row["spin_fragment4_final"] = sum(rec["spin_final"].get(i, 0.0) for i in fragment4)
        if rec["spin_initial"]:
            row["spin_fragment1_initial"] = sum(rec["spin_initial"].get(i, 0.0) for i in fragment1)
            row["spin_fragment4_initial"] = sum(rec["spin_initial"].get(i, 0.0) for i in fragment4)
        if rec["last_available_spin"]:
            row["spin_fragment1_last_available"] = sum(rec["last_available_spin"].get(i, 0.0) for i in fragment1)
            row["spin_fragment4_last_available"] = sum(rec["last_available_spin"].get(i, 0.0) for i in fragment4)

        # ---- Chemical integrity: BOTH initial and final geometry ----
        initial_atoms = None
        initial_broken_bonds, initial_clashes = set(), set()
        if geometry_file.exists():
            try:
                g0, initial_atoms = analyze_geometry(geometry_file, CENTER, left_ring, right_ring)
                for k, v in g0.items():
                    row[f"init_{k}"] = v
                chk0 = check_chemical_integrity(initial_atoms, bonds, bonded_pairs, BOND_STRETCH_TOL_ANG, BOND_ABS_MAX_ANG, MIN_NONBONDED_DIST_ANG)
                row["initial_geometry_valid"] = chk0["valid"]
                row["initial_min_nonbonded_distance"] = chk0["min_nonbonded_distance"]
                row["initial_n_broken_bonds"] = len(chk0["broken_bond_pairs"])
                row["initial_n_clashes"] = len(chk0["clash_pairs"])
                initial_broken_bonds, initial_clashes = chk0["broken_bond_pairs"], chk0["clash_pairs"]
            except Exception as exc:
                warnings.warn(f"{label}: failed to analyze initial geometry ({exc})")

        row["final_geometry_valid"] = None
        final_atoms = None
        if final_xyz.exists():
            try:
                g1, final_atoms = analyze_geometry(final_xyz, CENTER, left_ring, right_ring)
                for k, v in g1.items():
                    row[f"final_{k}"] = v
                chk1 = check_chemical_integrity(final_atoms, bonds, bonded_pairs, BOND_STRETCH_TOL_ANG, BOND_ABS_MAX_ANG, MIN_NONBONDED_DIST_ANG)
                row["final_geometry_valid"] = chk1["valid"]
                row["final_min_nonbonded_distance"] = chk1["min_nonbonded_distance"]
                row["chemical_break_reasons"] = "; ".join(chk1["reasons"])
                row["final_n_broken_bonds"] = len(chk1["broken_bond_pairs"])
                row["final_n_clashes"] = len(chk1["clash_pairs"])
                # Compare the initial and final broken-bond / clash SETS
                # independently: a bond that is newly broken (or a newly
                # created clash) counts as generated by the relaxation even if
                # the starting guess already had an unrelated clash elsewhere.
                new_broken = chk1["broken_bond_pairs"] - initial_broken_bonds
                new_clashes = chk1["clash_pairs"] - initial_clashes
                row["bond_broken_during_optimization"] = len(new_broken) > 0
                row["clash_created_during_optimization"] = len(new_clashes) > 0
                row["newly_broken_bonds"] = "; ".join(f"{a}-{b}" for a, b in sorted(new_broken))
                row["newly_created_clashes"] = "; ".join(f"{a}-{b}" for a, b in sorted(new_clashes))

                # Fragment 1-3 / 2-4 cross contacts (final geometry only)
                pos = final_atoms.get_positions()
                symbols = final_atoms.get_chemical_symbols()
                d13_all, pair13_all = cross_fragment_min_distance(pos, symbols, fragment1, fragment3, heavy_only=False)
                d24_all, pair24_all = cross_fragment_min_distance(pos, symbols, fragment2, fragment4, heavy_only=False)
                d13_heavy, pair13_heavy = cross_fragment_min_distance(pos, symbols, fragment1, fragment3, heavy_only=True)
                d24_heavy, pair24_heavy = cross_fragment_min_distance(pos, symbols, fragment2, fragment4, heavy_only=True)
                row["min_distance_1_3_all"] = d13_all
                row["min_distance_1_3_all_pair"] = str(pair13_all)
                row["min_distance_2_4_all"] = d24_all
                row["min_distance_2_4_all_pair"] = str(pair24_all)
                row["min_distance_1_3_heavy"] = d13_heavy
                row["min_distance_2_4_heavy"] = d24_heavy
                row["cross_distance_mean_heavy"] = (d13_heavy + d24_heavy) / 2.0

                # Robust packing descriptors (a single closest pair is fragile)
                m13, n13 = cross_fragment_contact_stats(pos, symbols, fragment1, fragment3, N_SHORTEST_CONTACTS, CONTACT_CUTOFF_ANG)
                m24, n24 = cross_fragment_contact_stats(pos, symbols, fragment2, fragment4, N_SHORTEST_CONTACTS, CONTACT_CUTOFF_ANG)
                row["mean_shortest_3_distance_1_3_heavy"] = m13
                row["mean_shortest_3_distance_2_4_heavy"] = m24
                row["n_contacts_below_4A_1_3_heavy"] = n13
                row["n_contacts_below_4A_2_4_heavy"] = n24
                row["cross_distance_robust_mean"] = (m13 + m24) / 2.0
                row["cross_contact_count_total"] = n13 + n24
            except Exception as exc:
                warnings.warn(f"{label}: failed to analyze final geometry ({exc})")

        row["angle_maintained"] = (
            "final_dihedral_actual" in row
            and abs(angle_diff_deg(row["final_dihedral_actual"], pt["C"])) <= ANGLE_TOLERANCE_DEG
        )
        row["usable"] = (row["status"] == "converged") and (row["final_geometry_valid"] is True)

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} points.")
    print(f"Convergence types: {df['convergence_type'].value_counts().to_dict()}")
    print(f"ORCA-converged (banner): {int((df['opt_converged'] == True).sum())}")
    print(f"Final geometry valid   : {int((df['final_geometry_valid'] == True).sum())}")
    print(f"Usable (converged AND valid): {int(df['usable'].sum())}")
    print()

    # ---- FOLDED / MAGNITUDE-HANDEDNESS TARGET QUANTITIES ----
    df["C_folded"] = df["C"].map(fold_to_0_90)
    df["plane_angle_deviation_final"] = df["final_central_plane_angle"] - df["C_folded"]
    df["plane_angle_deviation_initial"] = df["init_central_plane_angle"] - df["C_folded"]
    df["delta_tilt_ab"] = df["final_tilt_ab"] - df["init_tilt_ab"]
    df["delta_tilt_cd"] = df["final_tilt_cd"] - df["init_tilt_cd"]
    df["delta_bend_b"] = df["final_bend_b"] - df["init_bend_b"]
    df["delta_bend_c"] = df["final_bend_c"] - df["init_bend_c"]
    df["mean_terminal_spin_final"] = (df["spin_fragment1_final"] + df["spin_fragment4_final"]) / 2.0

    # ---- GLOBAL REFERENCE ENERGY ----
    usable = df["usable"]
    if usable.any():
        ref_idx = df.loc[usable, "final_energy"].idxmin()
        ref_row = df.loc[ref_idx]
        e_ref = ref_row["final_energy"]
        print(f"Global energy reference: {ref_row['label']} (E={e_ref:.6f} Eh)")
    else:
        e_ref = np.nan
        ref_row = None
        warnings.warn("No usable scan points -> relative energies undefined.")
    df["relative_energy_meV"] = np.where(df["final_energy"].notna(), (df["final_energy"] - e_ref) * HARTREE_TO_MEV, np.nan)

    # ---- ANOMALIES vs. NOTES ----
    # `flags` (anomalies) are reserved for: failed/unconverged jobs, broken or
    # clashing structures, constraint violations, anomalous <S^2>, and real
    # electronic peculiarities. Routine metadata (convergence type, starting
    # geometry strain, mild ring puckering, TRAH use) goes to `notes` and is
    # summarized statistically in the report instead of listed per point.
    flags = {label: [] for label in df["label"]}
    notes = {label: [] for label in df["label"]}
    for _, row in df.iterrows():
        label = row["label"]
        if row["status"] != "converged":
            flags[label].append(f"status={row['status']}")
        if row.get("convergence_type") not in ("strict", "not_converged"):
            failing = [
                f"{name}={crit['value']:.7f}(tol {crit['tolerance']:.7f})"
                for name, crit in (row.get("conv_criteria") or {}).items()
                if not crit["converged"]
            ]
            notes[label].append(f"convergence type = {row['convergence_type']}: " + "; ".join(failing))
        if row.get("bond_broken_during_optimization"):
            flags[label].append(f"bond(s) broken during optimization: {row.get('newly_broken_bonds', '')}")
        if row.get("clash_created_during_optimization"):
            flags[label].append(f"severe clash created during optimization: {row.get('newly_created_clashes', '')}")
        if row.get("initial_geometry_valid") is False:
            notes[label].append(
                f"starting geometry already had {int(row.get('initial_n_broken_bonds', 0))} stretched bond(s) / "
                f"{int(row.get('initial_n_clashes', 0))} close contact(s)"
            )
        if row.get("angle_maintained") is False:
            flags[label].append("constrained angle not maintained within tolerance")
        s2f = row.get("s2_final")
        if s2f is not None and pd.notna(s2f) and abs(s2f - IDEAL_S2_TRIPLET) > S2_ANOMALY_THRESHOLD:
            flags[label].append(f"final <S^2>={s2f:.3f} anomalous for a triplet")
        for ring_name in ("left", "right"):
            rms = row.get(f"final_{ring_name}_plane_rms")
            if rms is not None and pd.notna(rms) and rms > PLANARITY_RMS_THRESHOLD_ANG:
                notes[label].append(f"{ring_name} ring puckered (RMS={rms:.3f} Å)")
        if row.get("scf_not_converged"):
            notes[label].append("SCF NOT CONVERGED reported")
        if row.get("trah_activations", 0) > 0:
            notes[label].append(f"TRAH activated ({row['trah_activations']} times)")
        if row.get("neg_hessian_events", 0) > 0:
            notes[label].append(f"negative Hessian eigenvalues ({row['neg_hessian_events']} events)")

    # Electronic/branch-change anomaly: an energy discontinuity vs a grid
    # neighbor is only flagged if (a) <S^2> or terminal spin ALSO jumps along
    # the same neighbor pair AND (b) the two structures do NOT differ strongly
    # in plane-angle deviation, ring puckering or bond tilt -- otherwise the
    # energy/spin change has a structural explanation and there is no good
    # evidence the SCF moved to a different solution.
    grid = build_grid_index(df)
    axis_energy_steps, axis_s2_steps, axis_spin_steps = {}, {}, {}
    for axis in ("L", "C", "R"):
        e_diffs, s2_diffs, spin_diffs = [], [], []
        for idx, row in df.loc[usable].iterrows():
            for j in axis_neighbors(grid, row["L"], row["C"], row["R"], axis, df, usable_only=True):
                if j <= idx:
                    continue
                if pd.notna(row["relative_energy_meV"]) and pd.notna(df.loc[j, "relative_energy_meV"]):
                    e_diffs.append(abs(row["relative_energy_meV"] - df.loc[j, "relative_energy_meV"]))
                if pd.notna(row.get("s2_final")) and pd.notna(df.loc[j].get("s2_final")):
                    s2_diffs.append(abs(row["s2_final"] - df.loc[j, "s2_final"]))
                if pd.notna(row.get("mean_terminal_spin_final")) and pd.notna(df.loc[j].get("mean_terminal_spin_final")):
                    spin_diffs.append(abs(row["mean_terminal_spin_final"] - df.loc[j, "mean_terminal_spin_final"]))
        axis_energy_steps[axis] = max(ENERGY_DISCONTINUITY_FACTOR * (float(np.median(e_diffs)) if e_diffs else 0.0), ENERGY_DISCONTINUITY_FLOOR_MEV)
        axis_s2_steps[axis] = max(3.0 * (float(np.median(s2_diffs)) if s2_diffs else 0.0), S2_DISCONTINUITY_FLOOR)
        axis_spin_steps[axis] = max(3.0 * (float(np.median(spin_diffs)) if spin_diffs else 0.0), SPIN_DISCONTINUITY_FLOOR)

    def structural_jump(a, b):
        """True if points a, b differ strongly in accommodation descriptors."""
        def diff(col):
            va, vb = a.get(col), b.get(col)
            return abs(va - vb) if (pd.notna(va) and pd.notna(vb)) else 0.0

        mean_rms_a = np.nanmean([a.get("final_left_plane_rms"), a.get("final_right_plane_rms")])
        mean_rms_b = np.nanmean([b.get("final_left_plane_rms"), b.get("final_right_plane_rms")])
        return (
            diff("plane_angle_deviation_final") > STRUCTURAL_JUMP_PLANE_ANGLE_DEG
            or abs(mean_rms_a - mean_rms_b) > STRUCTURAL_JUMP_PLANARITY_RMS_ANG
            or diff("final_tilt_ab") > STRUCTURAL_JUMP_TILT_DEG
            or diff("final_tilt_cd") > STRUCTURAL_JUMP_TILT_DEG
        )

    for idx, row in df.iterrows():
        if not row["usable"]:
            continue
        label = row["label"]
        for axis in ("L", "C", "R"):
            neighbor_idxs = axis_neighbors(grid, row["L"], row["C"], row["R"], axis, df, usable_only=True)
            for j in neighbor_idxs:
                nrow = df.loc[j]
                e_jump = pd.notna(row["relative_energy_meV"]) and pd.notna(nrow["relative_energy_meV"]) and abs(row["relative_energy_meV"] - nrow["relative_energy_meV"]) > axis_energy_steps[axis]
                s2_jump = pd.notna(row.get("s2_final")) and pd.notna(nrow.get("s2_final")) and abs(row["s2_final"] - nrow["s2_final"]) > axis_s2_steps[axis]
                spin_jump = pd.notna(row.get("mean_terminal_spin_final")) and pd.notna(nrow.get("mean_terminal_spin_final")) and abs(row["mean_terminal_spin_final"] - nrow["mean_terminal_spin_final"]) > axis_spin_steps[axis]
                if not (e_jump and (s2_jump or spin_jump)):
                    continue
                if structural_jump(row, nrow):
                    notes[label].append(
                        f"energy+spin jump vs {nrow['label']} accompanied by a large structural change "
                        f"(not attributed to an SCF branch change)"
                    )
                else:
                    flags[label].append(
                        f"possible electronic/SCF branch change vs {nrow['label']} "
                        f"(ΔE={row['relative_energy_meV']-nrow['relative_energy_meV']:.0f} meV, "
                        f"{'Δ<S^2>' if s2_jump else ''}{' & ' if s2_jump and spin_jump else ''}{'Δspin' if spin_jump else ''} also jumps, "
                        f"no comparable structural change)"
                    )
                break

        # Spin-topology anomaly relative to grid neighbors
        if row.get("spin_fragment1_final") is not None and row.get("spin_fragment4_final") is not None and pd.notna(row["spin_fragment1_final"]):
            this_pattern = (np.sign(row["spin_fragment1_final"]), np.sign(row["spin_fragment4_final"]))
            neighbor_patterns = []
            for j in grid_neighbors(grid, row["L"], row["C"], row["R"], df, usable_only=True):
                nl_, nr_ = df.loc[j, "spin_fragment1_final"], df.loc[j, "spin_fragment4_final"]
                if pd.notna(nl_) and pd.notna(nr_):
                    neighbor_patterns.append((np.sign(nl_), np.sign(nr_)))
            if len(neighbor_patterns) >= 2 and len(set(neighbor_patterns)) == 1 and this_pattern != neighbor_patterns[0]:
                flags[label].append("unusual spin topology vs grid neighbors (possible branch switch)")

    df["flags"] = df["label"].map(lambda lbl: "; ".join(dict.fromkeys(flags[lbl])) if flags[lbl] else "")
    df["n_flags"] = df["label"].map(lambda lbl: len(set(flags[lbl])))
    df["notes"] = df["label"].map(lambda lbl: "; ".join(dict.fromkeys(notes[lbl])) if notes[lbl] else "")

    # ---- SAVE CSV ----
    csv_df = df.copy()
    for atom in RADICAL_ATOMS:
        for kind in ("initial", "final", "last_available"):
            col = f"radical_spin_{kind}"
            csv_df[f"{col}_{atom}"] = df[col].map(lambda dd, a=atom: dd.get(a))
    csv_df = csv_df.drop(
        columns=["radical_spin_initial", "radical_spin_final", "radical_spin_last_available", "spin_initial", "last_available_spin", "conv_criteria", "label"],
        errors="ignore",
    )
    csv_df.insert(0, "label", df["label"])
    csv_df.to_csv(CSV_PATH, index=False)
    print(f"Wrote {CSV_PATH}")

    # ---- PBE0 HS/BS SCAFFOLD (populates only if data exists) ----
    df = attach_pbe0_hsbs(df, points, fragment1, fragment4)

    # ---- PLOTS ----
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
    free_relax = load_free_relaxation(ROOT_DIR.parent / "angle_scan_free_relaxation", CENTER, e_ref)
    print(f"Free-relaxation results found: {len(free_relax)}")
    make_plots(df, PLOTS_DIR, HEATMAPS_DIR, RADICAL_ATOMS, free_relax)
    print(f"Wrote plots to {PLOTS_DIR}")

    # ---- MARKDOWN REPORT ----
    write_markdown_report(df, MD_PATH, CENTER, LEFT, RIGHT, left_ring, right_ring, fragment1, fragment4, ROOT_DIR, ref_row, free_relax)
    print(f"Wrote {MD_PATH}")

    print_summary(df)


def build_grid_index(df):
    index_map = {(row.L, row.C, row.R): idx for idx, row in zip(df.index, df[["L", "C", "R"]].itertuples(index=False))}
    axis_values = {axis: sorted(df[axis].unique()) for axis in ("L", "C", "R")}
    return {"index_map": index_map, "axis_values": axis_values}


def axis_neighbors(grid, L, C, R, axis, df, usable_only=True):
    axis_values = grid["axis_values"][axis]
    index_map = grid["index_map"]
    current = {"L": L, "C": C, "R": R}
    pos = axis_values.index(current[axis])
    neighbor_idxs = []
    for step in (-1, 1):
        npos = pos + step
        if 0 <= npos < len(axis_values):
            key = dict(current)
            key[axis] = axis_values[npos]
            idx = index_map.get((key["L"], key["C"], key["R"]))
            if idx is not None and (not usable_only or df.loc[idx, "usable"]):
                neighbor_idxs.append(idx)
    return neighbor_idxs


def grid_neighbors(grid, L, C, R, df, usable_only=True):
    neighbor_idxs = []
    for axis in ("L", "C", "R"):
        neighbor_idxs.extend(axis_neighbors(grid, L, C, R, axis, df, usable_only=usable_only))
    return neighbor_idxs


def attach_pbe0_hsbs(df, points, fragment1, fragment4):
    """
    Optionally detect <folder>/high_spin/high_spin.out and
    <folder>/broken_symmetry/broken_symmetry.out for each scan point (a
    separate PBE0 single-point HS/BS exchange calculation on the frozen
    R2SCAN-3C-relaxed geometry). Columns are NaN until that data exists.
    """
    by_label = {pt["folder"].name: pt["folder"] for pt in points}
    e_hs, e_bs, s2_hs, s2_bs, spin1_hs, spin4_hs, spin1_bs, spin4_bs = ({} for _ in range(8))

    for label, folder in by_label.items():
        hs_out = folder / "high_spin" / "high_spin.out"
        bs_out = folder / "broken_symmetry" / "broken_symmetry.out"
        if not (hs_out.exists() and bs_out.exists()):
            continue
        hs_text = hs_out.read_text(errors="ignore")
        bs_text = bs_out.read_text(errors="ignore")
        hs_e = FINAL_E_RE.findall(hs_text)
        bs_e = FINAL_E_RE.findall(bs_text)
        if not (hs_e and bs_e):
            continue
        e_hs[label], e_bs[label] = float(hs_e[-1]), float(bs_e[-1])
        hs_s2, bs_s2 = S2_RE.findall(hs_text), S2_RE.findall(bs_text)
        s2_hs[label] = float(hs_s2[-1]) if hs_s2 else None
        s2_bs[label] = float(bs_s2[-1]) if bs_s2 else None

        for text, spin1_map, spin4_map in ((hs_text, spin1_hs, spin4_hs), (bs_text, spin1_bs, spin4_bs)):
            blocks = list(MULLIKEN_BLOCK_RE.finditer(text))
            if not blocks:
                continue
            spins = {}
            for line in blocks[-1].group(1).splitlines():
                m = ATOM_LINE_RE.match(line)
                if m:
                    spins[int(m.group(1))] = float(m.group(4))
            spin1_map[label] = sum(spins.get(i, 0.0) for i in fragment1)
            spin4_map[label] = sum(spins.get(i, 0.0) for i in fragment4)

    df["E_HS_pbe0"] = df["label"].map(e_hs)
    df["E_BS_pbe0"] = df["label"].map(e_bs)
    df["delta_E_exchange_meV"] = (df["E_HS_pbe0"] - df["E_BS_pbe0"]) * HARTREE_TO_MEV
    df["s2_HS_pbe0"] = df["label"].map(s2_hs)
    df["s2_BS_pbe0"] = df["label"].map(s2_bs)
    df["spin_fragment1_HS_pbe0"] = df["label"].map(spin1_hs)
    df["spin_fragment4_HS_pbe0"] = df["label"].map(spin4_hs)
    df["spin_fragment1_BS_pbe0"] = df["label"].map(spin1_bs)
    df["spin_fragment4_BS_pbe0"] = df["label"].map(spin4_bs)
    n_found = df["E_HS_pbe0"].notna().sum()
    print(f"PBE0 HS/BS pairs found: {n_found} (of {len(df)} scan points)")
    return df


def make_plots(df, plots_dir, heatmaps_dir, radical_atoms, free_relax=()):
    usable = df["usable"]
    not_usable = ~usable

    def scatter_by_status(ax, x_col, y_col):
        ax.scatter(df.loc[usable, x_col], df.loc[usable, y_col], marker="o", color="tab:blue", label="usable")
        ax.scatter(df.loc[not_usable, x_col], df.loc[not_usable, y_col], marker="x", color="tab:red", label="not usable")

    c_values = sorted(df["C"].unique())
    l_values = sorted(df["L"].unique())
    r_values = sorted(df["R"].unique())

    # --- Convergence fraction vs C ---
    fig, ax = plt.subplots()
    n_conv = [((df["C"] == cv) & (df["status"] == "converged")).sum() for cv in c_values]
    n_tot = [(df["C"] == cv).sum() for cv in c_values]
    ax.bar([str(int(cv)) for cv in c_values], n_conv, color="tab:blue", label="converged")
    ax.bar([str(int(cv)) for cv in c_values], [t - n for t, n in zip(n_tot, n_conv)], bottom=n_conv, color="tab:red", alpha=0.6, label="not converged/failed")
    ax.set_xlabel("Central (C) target dihedral (deg)")
    ax.set_ylabel("Number of (L,R) combinations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "convergence_fraction_vs_C.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    scatter_by_status(ax, "C", "n_geom_cycles")
    ax.set_xlabel("Central (C) target dihedral (deg)")
    ax.set_ylabel("Geometry optimization cycles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "optimization_cycles_vs_C.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    scatter_by_status(ax, "C", "final_min_nonbonded_distance")
    ax.set_xlabel("Central (C) target dihedral (deg)")
    ax.set_ylabel("Min. non-bonded distance, final structure (Å)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "steric_vs_C.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    scatter_by_status(ax, "C", "relative_energy_meV")
    ax.set_xlabel("Central (C) target dihedral (deg)")
    ax.set_ylabel("Relative energy (meV)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "relative_energy_vs_C_all_LR.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    scatter_by_status(ax, "C", "s2_final")
    ax.axhline(2.0, color="k", linestyle=":", linewidth=1, label="ideal triplet")
    ax.set_xlabel("Central (C) target dihedral (deg)")
    ax.set_ylabel("Final <S^2>")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "s2_vs_C.png", dpi=150)
    plt.close(fig)

    # --- Central plane angle: mean+range vs C, and vs target ---
    fig, ax = plt.subplots()
    lims = [0, 95]
    ax.plot(lims, lims, "k--", linewidth=1, label="plane angle = target (folded)")
    ax.scatter(df["C_folded"], df["init_central_plane_angle"], marker="s", facecolors="none", edgecolors="tab:grey", label="initial")
    scatter_by_status(ax, "C_folded", "final_central_plane_angle")
    ax.set_xlabel("Constrained CENTER dihedral, folded to 0-90 deg")
    ax.set_ylabel("Fitted central-plane angle (deg)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "plane_angle_vs_target.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    means = [df.loc[usable & (df["C"] == cv), "final_central_plane_angle"].mean() for cv in c_values]
    stds = [df.loc[usable & (df["C"] == cv), "final_central_plane_angle"].std() for cv in c_values]
    ax.errorbar(c_values, means, yerr=stds, marker="o", capsize=3, label="mean ± std over all L,R")
    ax.plot(c_values, [fold_to_0_90(cv) for cv in c_values], "k--", linewidth=1, label="target (folded)")
    ax.set_xlabel("Central (C) target dihedral (deg)")
    ax.set_ylabel("Central plane angle (deg)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "mean_plane_angle_vs_C.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.axhline(0, color="k", linewidth=1)
    scatter_by_status(ax, "C", "plane_angle_deviation_final")
    ax.set_xlabel("Central (C) target dihedral (deg)")
    ax.set_ylabel("Plane angle − target (deg)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "plane_angle_deviation_vs_C.png", dpi=150)
    plt.close(fig)

    # --- Bond tilt and bending accommodation vs C ---
    fig, ax = plt.subplots()
    ax.axhline(0, color="k", linewidth=1)
    ax.scatter(df.loc[usable, "C"], df.loc[usable, "delta_tilt_ab"], marker="o", label="Δtilt A-B")
    ax.scatter(df.loc[usable, "C"], df.loc[usable, "delta_tilt_cd"], marker="^", label="Δtilt C-D")
    ax.set_xlabel("Central (C) target dihedral (deg)")
    ax.set_ylabel("Δ bond tilt, final − initial (deg)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "bond_tilt_vs_C.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.axhline(0, color="k", linewidth=1)
    ax.scatter(df.loc[usable, "C"], df.loc[usable, "delta_bend_b"], marker="o", label="pivot B puckering, Δ vs initial")
    ax.scatter(df.loc[usable, "C"], df.loc[usable, "delta_bend_c"], marker="^", label="pivot C puckering, Δ vs initial")
    ax.set_xlabel("Central (C) target dihedral (deg)")
    ax.set_ylabel("Δ out-of-plane displacement (Å)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "bending_accommodation_vs_C.png", dpi=150)
    plt.close(fig)

    # --- Fragment spin vs outer torsion, with std/range ---
    outer_vals = sorted(set(df["L"].unique()) | set(df["R"].unique()))
    fig, ax = plt.subplots()
    for frag, axis, color in (("spin_fragment1_final", "L", "tab:blue"), ("spin_fragment4_final", "R", "tab:orange")):
        means = [df.loc[usable & (df[axis] == v), frag].mean() for v in outer_vals]
        stds = [df.loc[usable & (df[axis] == v), frag].std() for v in outer_vals]
        ax.errorbar(outer_vals, means, yerr=stds, marker="o", capsize=3, color=color, label=f"fragment {'1 (vs L)' if axis=='L' else '4 (vs R)'}")
    ax.set_xlabel("Outer (L or R) target dihedral (deg)")
    ax.set_ylabel("Region-integrated final spin population")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "fragment_spin_vs_outer_torsion.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    means = [df.loc[usable & (df["C"] == cv), "spin_fragment2_final"].mean() + df.loc[usable & (df["C"] == cv), "spin_fragment3_final"].mean() for cv in c_values]
    ax.plot(c_values, means, "o-")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Central (C) target dihedral (deg)")
    ax.set_ylabel("Mean central-backbone spin (fragments 2+3)")
    fig.tight_layout()
    fig.savefig(plots_dir / "central_backbone_spin_vs_C.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    for atom in radical_atoms:
        vals = df["radical_spin_final"].map(lambda d, a=atom: d.get(a))
        means = [vals.loc[usable & (df["L"] == v)].mean() for v in outer_vals]
        ax.plot(outer_vals, means, "o-", label=f"atom {atom} (vs L)")
    ax.set_xlabel("Outer (L) target dihedral (deg)")
    ax.set_ylabel("Mean final Mulliken spin population")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "radical_atom_spin_vs_L_supplementary.png", dpi=150)
    plt.close(fig)

    # --- <S^2> vs mean terminal spin localization ---
    fig, ax = plt.subplots()
    ax.scatter(df.loc[usable, "mean_terminal_spin_final"], df.loc[usable, "s2_final"], marker="o")
    r, n = pearson_r(df.loc[usable, "mean_terminal_spin_final"], df.loc[usable, "s2_final"])
    ax.set_xlabel("Mean terminal-fragment spin, (frag1+frag4)/2")
    ax.set_ylabel("Final <S^2>")
    ax.set_title(f"Pearson r = {r:.2f} (n={n})")
    fig.tight_layout()
    fig.savefig(plots_dir / "s2_vs_terminal_spin_localization.png", dpi=150)
    plt.close(fig)

    # --- Per-C energy heatmaps over (L, R) ---
    vmin, vmax = df.loc[usable, "relative_energy_meV"].min(), df.loc[usable, "relative_energy_meV"].max()
    for cv in c_values:
        grid_e = np.full((len(l_values), len(r_values)), np.nan)
        for _, row in df[df["C"] == cv].iterrows():
            if not row["usable"]:
                continue
            grid_e[l_values.index(row["L"]), r_values.index(row["R"])] = row["relative_energy_meV"]
        fig, ax = plt.subplots()
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="lightgrey")
        im = ax.imshow(np.ma.masked_invalid(grid_e), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(r_values)))
        ax.set_xticklabels([f"{int(v)}" for v in r_values])
        ax.set_yticks(range(len(l_values)))
        ax.set_yticklabels([f"{int(v)}" for v in l_values])
        ax.set_xlabel("R (deg)")
        ax.set_ylabel("L (deg)")
        ax.set_title(f"Relative energy (meV), C={int(cv)}°")
        fig.colorbar(im, ax=ax, label="meV")
        fig.tight_layout()
        fig.savefig(heatmaps_dir / f"energy_heatmap_C{int(cv):03d}.png", dpi=150)
        plt.close(fig)

    # --- 45/45 vs 135/135 crossover ---
    make_crossover_plots(df, plots_dir, c_values)
    make_headline_plots(df, plots_dir, c_values, list(free_relax))

    # --- PBE0 HS/BS plots (only if data present) ---
    if df["E_HS_pbe0"].notna().any():
        make_pbe0_plots(df, plots_dir)


def symmetric_subset(df, l_val, r_val):
    return df[(df["L"] == l_val) & (df["R"] == r_val) & df["usable"]].set_index("C")


def crossover_descriptors():
    """(column, label) of the cross-fragment packing descriptors compared in the crossover."""
    return [
        ("cross_distance_mean_heavy", "closest-pair distance (mean of 1-3, 2-4)"),
        ("cross_distance_robust_mean", "mean of 3 shortest heavy-atom contacts (mean of 1-3, 2-4)"),
    ]


def make_crossover_plots(df, plots_dir, c_values):
    sym45 = symmetric_subset(df, 45.0, 45.0)
    sym135 = symmetric_subset(df, 135.0, 135.0)
    common_c = sorted(set(sym45.index) & set(sym135.index))
    if not common_c:
        return

    delta_e = [sym45.loc[cv, "relative_energy_meV"] - sym135.loc[cv, "relative_energy_meV"] for cv in common_c]
    descriptors = crossover_descriptors()

    # Two-panel main figure: dE(C) on top, d(cross)(C) below, zero line in both.
    fig, (ax_e, ax_d) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 7))
    ax_e.axhline(0, color="k", linewidth=1)
    ax_e.plot(common_c, delta_e, "o-", color="tab:purple")
    ax_e.set_ylabel("ΔE = E(45/45) − E(135/135) (meV)")
    ax_e.set_title("Positive: 135/135 lower in energy;  negative: 45/45 lower")

    ax_d.axhline(0, color="k", linewidth=1)
    styles = [("o-", "tab:blue"), ("s--", "tab:green")]
    for (col, label), (fmt, color) in zip(descriptors, styles):
        delta_d = [sym45.loc[cv, col] - sym135.loc[cv, col] for cv in common_c]
        ax_d.plot(common_c, delta_d, fmt, color=color, label=label)
    ax_d.set_xlabel("Central (C) target dihedral (deg)")
    ax_d.set_ylabel("Δd_cross = d(45/45) − d(135/135) (Å)")
    ax_d.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "crossover_two_panel.png", dpi=150)
    plt.close(fig)

    # Supplementary scatter: dE vs dd for each descriptor, with Pearson r.
    fig, axes = plt.subplots(1, len(descriptors), figsize=(6.2 * len(descriptors) / 1.4, 4.6))
    for ax, (col, label) in zip(np.atleast_1d(axes), descriptors):
        delta_d = [sym45.loc[cv, col] - sym135.loc[cv, col] for cv in common_c]
        ax.scatter(delta_d, delta_e)
        for cv, x, y in zip(common_c, delta_d, delta_e):
            ax.annotate(f"{int(cv)}", (x, y), fontsize=7)
        r, n = pearson_r(delta_d, delta_e)
        ax.set_xlabel("Δd_cross (Å)")
        ax.set_ylabel("ΔE (meV)")
        ax.set_title(f"{label}\nPearson r = {r:.2f} (n={n})", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "delta_E_vs_delta_cross_contact.png", dpi=150)
    plt.close(fig)


def load_free_relaxation(free_root, center, e_ref):
    """
    Optional overlay: unconstrained HS/BS free relaxations (angle_scan_free_relaxation).
    Returns [{label, state, central_dihedral, rel_energy_meV, converged}] for every
    <point>/<state>/<state>.out that exists; empty list if none.
    """
    results = []
    if not free_root.exists() or not np.isfinite(e_ref):
        return results
    for point in sorted(p for p in free_root.iterdir() if p.is_dir()):
        for state in ("high_spin", "broken_symmetry"):
            out_file = point / state / f"{state}.out"
            xyz_file = point / state / f"{state}.xyz"
            if not out_file.exists():
                continue
            text = out_file.read_text(errors="ignore")
            energies = FINAL_E_RE.findall(text)
            if not energies:
                continue
            converged, _, _ = classify_convergence(text)
            entry = {
                "label": point.name,
                "state": state,
                "rel_energy_meV": (float(energies[-1]) - e_ref) * HARTREE_TO_MEV,
                "converged": converged,
                "central_dihedral": np.nan,
            }
            if xyz_file.exists():
                try:
                    entry["central_dihedral"] = float(read(xyz_file).get_dihedral(*center))
                except Exception:
                    pass
            results.append(entry)
    return results


def make_headline_plots(df, plots_dir, c_values, free_relax):
    usable = df["usable"]

    # --- Principal conformational-energy figure: E(C) for symmetric outer
    # torsions + E_min(C) = min over (L, R), plus optional free-relaxation markers.
    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {0.0: ("tab:blue", "-"), 45.0: ("tab:orange", "-"), 90.0: ("tab:green", "-"),
              135.0: ("tab:red", "-"), 180.0: ("tab:grey", "--")}
    for outer, (color, ls) in styles.items():
        sub = df[usable & (df["L"] == outer) & (df["R"] == outer)].sort_values("C")
        if sub.empty:
            continue
        ax.plot(sub["C"], sub["relative_energy_meV"], marker="o", ls=ls, color=color,
                alpha=0.55 if outer == 180.0 else 0.9, label=f"L=R={int(outer)}°")

    e_min = df[usable].groupby("C")["relative_energy_meV"].min()
    ax.plot(e_min.index, e_min.values, "k-", linewidth=3, marker="o", label="E_min(C) = min over (L,R)")

    for entry in free_relax:
        if not np.isfinite(entry["central_dihedral"]):
            continue
        marker = "*" if entry["state"] == "high_spin" else "D"
        ax.scatter(entry["central_dihedral"], entry["rel_energy_meV"], marker=marker, s=130,
                   facecolors="none" if not entry["converged"] else "gold", edgecolors="k", zorder=6)
    if free_relax:
        ax.scatter([], [], marker="*", s=130, facecolors="gold", edgecolors="k", label="free HS relaxation")
        ax.scatter([], [], marker="D", s=80, facecolors="gold", edgecolors="k", label="free BS relaxation")

    ax.set_xlabel("Central (C) dihedral (deg)")
    ax.set_ylabel("Relative energy (meV), R2SCAN-3C HS surface")
    ax.set_title("Conformational energy vs central torsion")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "energy_vs_C_symmetric_outer.png", dpi=150)
    plt.close(fig)

    # --- Radical localization vs OUTER angle, L and R pooled (mean +/- std) ---
    outer_vals = sorted(set(df["L"].unique()) | set(df["R"].unique()))
    pooled_mean, pooled_std, pooled_n = [], [], []
    for v in outer_vals:
        vals = pd.concat([
            df.loc[usable & (df["L"] == v), "spin_fragment1_final"],
            df.loc[usable & (df["R"] == v), "spin_fragment4_final"],
        ]).dropna()
        pooled_mean.append(vals.mean())
        pooled_std.append(vals.std())
        pooled_n.append(len(vals))
    fig, ax = plt.subplots()
    ax.errorbar(outer_vals, pooled_mean, yerr=pooled_std, marker="o", capsize=4, color="tab:blue")
    ax.set_xlabel("Outer dihedral, own side (L for fragment 1, R for fragment 4) (deg)")
    ax.set_ylabel("Terminal-fragment spin population (mean ± std)")
    ax.set_title("Radical localization is set by its own outer torsion")
    fig.tight_layout()
    fig.savefig(plots_dir / "terminal_spin_vs_outer_pooled.png", dpi=150)
    plt.close(fig)

    # --- Control: localization vs C at fixed L=R=45 (expected ~flat) ---
    sub = df[usable & (df["L"] == 45.0) & (df["R"] == 45.0)].sort_values("C")
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(sub["C"], sub["spin_fragment1_final"], "o-", label="fragment 1")
        ax.plot(sub["C"], sub["spin_fragment4_final"], "s--", label="fragment 4")
        ax.set_xlabel("Central (C) dihedral (deg)")
        ax.set_ylabel("Terminal-fragment spin")
        ax.set_title("Control: L=R=45° fixed, C varied")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / "terminal_spin_vs_C_control_L45R45.png", dpi=150)
        plt.close(fig)


def make_pbe0_plots(df, plots_dir):
    have = df["E_HS_pbe0"].notna() & df["usable"]
    fig, ax = plt.subplots()
    ax.scatter(df.loc[have, "C"], df.loc[have, "delta_E_exchange_meV"])
    ax.axhline(0, color="k", linewidth=1)
    ax.set_xlabel("Central (C) target dihedral (deg)")
    ax.set_ylabel("ΔE_exchange = E_HS − E_BS (meV, raw)")
    fig.tight_layout()
    fig.savefig(plots_dir / "exchange_vs_C.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(df.loc[have, "mean_terminal_spin_final"], df.loc[have, "delta_E_exchange_meV"])
    ax.axhline(0, color="k", linewidth=1)
    r, n = pearson_r(df.loc[have, "mean_terminal_spin_final"], df.loc[have, "delta_E_exchange_meV"])
    ax.set_xlabel("Mean terminal-fragment spin localization (R2SCAN-3C)")
    ax.set_ylabel("ΔE_exchange (meV, raw)")
    ax.set_title(f"Pearson r = {r:.2f} (n={n})")
    fig.tight_layout()
    fig.savefig(plots_dir / "exchange_vs_terminal_spin.png", dpi=150)
    plt.close(fig)


def write_markdown_report(df, md_path, center, left, right, left_ring, right_ring, fragment1, fragment4, root_dir, ref_row, free_relax=()):
    lines = []
    usable = df["usable"]
    lines.append("# ORCA constrained-torsion angle scan — full L×C×R grid\n")

    lines.append("## Summary of the picture\n")
    lines.append(
        "1. **Central torsion → conformational energy and structural strain.**\n"
        "2. **Outer torsions → radical localization** (each terminal radical is controlled by its own outer torsion).\n"
        "3. **Radical localization → exchange coupling J**: the third step, to be provided by the separate PBE0 HS/BS "
        "single points (not yet available; see the last section).\n"
    )

    lines.append("## Overview and methods\n")
    lines.append(f"- Scan root: `{root_dir}`")
    lines.append(f"- {len(df)} scan points, R2SCAN-3C TightOpt UKS, high-spin (multiplicity 3)")
    lines.append(f"- CENTER dihedral {center} (bond {center[1]}-{center[2]}), LEFT {left}, RIGHT {right}")
    lines.append(f"- Fragment 1 (LEFT terminal, radical atom 41, {len(fragment1)} atoms): {fragment1}")
    lines.append(f"- Fragment 4 (RIGHT terminal, radical atom 28, {len(fragment4)} atoms): {fragment4}")
    lines.append("- Fragments 2/3: the central backbone, split at the CENTER bond")
    lines.append(
        "- **Convergence** is read from ORCA's literal `THE OPTIMIZATION HAS CONVERGED` banner (authoritative); "
        "the override type ORCA used is recorded separately and summarized statistically below (it is metadata, not an anomaly)."
    )
    lines.append(
        "- **Chemical integrity**: the SETS of broken reference bonds (stretch > 0.30 Å or > 1.9 Å) and severe "
        "non-bonded clashes (< 1.5 Å) are computed for the initial and the final geometry independently. A bond or "
        "clash present in the final set but not the initial set is attributed to the relaxation — an unrelated clash "
        "in the starting guess never masks it."
    )
    lines.append(
        "- **Spin populations**: `spin_final`/`s2_final` require BOTH ORCA convergence AND a post-optimization Mulliken "
        "analysis. Unconverged jobs never enter primary plots; their last available (initial-SCF) spin is kept only as "
        "`*_last_available_*` bookkeeping."
    )
    lines.append(
        "- **Anomalies vs notes**: the anomaly list is reserved for failed/unconverged jobs, chemically broken structures, "
        "constraint violations, anomalous ⟨S²⟩ and electronic peculiarities. Routine metadata (convergence "
        "type, strained starting guess, ring puckering, TRAH use) is in the `notes` column of `angle_scan.csv`. A large "
        "energy change between neighboring grid points is NOT an anomaly by itself; it is flagged as a possible "
        "SCF branch change only if ⟨S²⟩/spin also jump AND the two structures do not differ strongly in "
        "plane-angle deviation, ring puckering or bond tilt (otherwise the change has a structural explanation)."
    )
    lines.append(
        "- **Magnitude/handedness**: `magnitude = min(θ, 180−θ)`, `handedness = sign(θ−90)`; 45° and "
        "135° share magnitude but have opposite handedness and are never treated as equivalent."
    )
    lines.append("")

    if (md_path.parent / "plots" / "torsion_scheme.png").exists():
        lines.append("**What the scan does** (generated by `draw_torsion_scheme.py`): the three constrained dihedrals L, C, R and the four fragments 1\u20132\u20133\u20134.\n")
        lines.append("![Torsion scheme](plots/torsion_scheme.png)\n")

    # ---- Q1 ----
    lines.append("## Question 1 — Which L,C,R space is accessible?\n")
    lines.append("| C (deg) | ORCA-converged / 25 | final geometry valid | usable |")
    lines.append("|---|---|---|---|")
    for cv in sorted(df["C"].unique()):
        sub = df[df["C"] == cv]
        lines.append(f"| {int(cv)} | {(sub['status']=='converged').sum()}/{len(sub)} | {(sub['final_geometry_valid']==True).sum()} | {sub['usable'].sum()} |")
    lines.append("")
    lines.append("![Convergence vs C](plots/convergence_fraction_vs_C.png)\n")
    lines.append("![Optimization cycles vs C](plots/optimization_cycles_vs_C.png)\n")
    lines.append("![Steric vs C](plots/steric_vs_C.png)\n")

    conv_df = df[df["status"] == "converged"]
    lines.append("**Convergence types among the ORCA-converged points** (routine metadata):\n")
    type_counts = conv_df["convergence_type"].value_counts()
    lines.append("| type | count | fraction |")
    lines.append("|---|---|---|")
    for t, cnt in type_counts.items():
        lines.append(f"| {t} | {cnt} | {cnt/len(conv_df):.0%} |")
    lines.append("")
    n_fail = int((df["status"] != "converged").sum())
    fail_counts = df.loc[df["status"] != "converged", "status"].value_counts().to_dict()
    lines.append(f"Non-converged: {n_fail} points ({fail_counts}).\n")

    broken_converged = df[(df["status"] == "converged") & (df["final_geometry_valid"] == False)]
    lines.append(f"**Converged-but-chemically-broken points** (ORCA's convergence signal does not catch these): {len(broken_converged)}\n")
    for _, row in broken_converged.iterrows():
        lines.append(f"- **{row['label']}**: E={row['final_energy']:.6f} Eh; newly broken bonds {row.get('newly_broken_bonds', '')}; {row['chemical_break_reasons']}")
    lines.append("")

    # ---- Q2 / headline energy ----
    lines.append("## Question 2 — Central torsion → conformational energy and structural accommodation\n")
    lines.append(
        "**Main figure.** E(C) for symmetric outer torsions L=R=0/45/90/135° (180° dashed, lighter), and the thick "
        "black curve E_min(C) = min over (L,R): the best energy the molecule reaches at each central angle after choosing "
        "its outer conformation, which separates the intrinsic preference for C from an arbitrary choice of fixed outer "
        "angle. "
        + ("Free HS/BS relaxations found are overlaid as markers (gold = converged).\n" if free_relax else
           "Free HS/BS relaxations (`angle_scan_free_relaxation`) will be overlaid as markers once their outputs exist.\n")
    )
    lines.append("![E(C) symmetric outer](plots/energy_vs_C_symmetric_outer.png)\n")
    e_min = df[usable].groupby("C")["relative_energy_meV"].min()
    lines.append("| C (deg) | E_min(C) (meV above global min) |")
    lines.append("|---|---|")
    for cv, e in e_min.items():
        lines.append(f"| {int(cv)} | {e:.0f} |")
    lines.append("")
    if ref_row is not None:
        lines.append(f"Global reference: **{ref_row['label']}** (L={ref_row['L']:.0f}, C={ref_row['C']:.0f}, R={ref_row['R']:.0f}), E={ref_row['final_energy']:.8f} Eh.\n")

    lines.append("### Structural accommodation of the imposed central torsion\n")
    lines.append(
        "ORCA holds the literal 4-atom CENTER dihedral exactly at target, but the fitted planes of the two flanking "
        "aromatic rings can still separate by a different angle: the molecule distorts internally (ring puckering, bond "
        "tilt) so that the actual π planes do not simply follow the imposed dihedral. y = x is the rigid-fragment expectation.\n"
    )
    lines.append("![Plane angle vs target](plots/plane_angle_vs_target.png)\n")
    lines.append("![Mean plane angle vs C](plots/mean_plane_angle_vs_C.png)\n")
    lines.append("![Plane angle deviation vs C](plots/plane_angle_deviation_vs_C.png)\n")
    lines.append("![Bond tilt vs C](plots/bond_tilt_vs_C.png)\n")
    lines.append("![Puckering vs C](plots/bending_accommodation_vs_C.png)\n")

    # ---- Q3/Q4 ----
    lines.append("## Questions 3 & 4 — The 45/45 ↔ 135/135 crossover and 1–3 / 2–4 packing\n")
    sym45 = symmetric_subset(df, 45.0, 45.0)
    sym135 = symmetric_subset(df, 135.0, 135.0)
    common_c = sorted(set(sym45.index) & set(sym135.index))
    if common_c:
        lines.append("| C | E(45/45) (meV) | E(135/135) (meV) | preferred | Δd closest-pair (Å) | Δd 3-shortest mean (Å) | Δ #contacts<4Å |")
        lines.append("|---|---|---|---|---|---|---|")
        for cv in common_c:
            e45, e135 = sym45.loc[cv, "relative_energy_meV"], sym135.loc[cv, "relative_energy_meV"]
            pref = "45/45" if e45 < e135 else "135/135"
            d_min = sym45.loc[cv, "cross_distance_mean_heavy"] - sym135.loc[cv, "cross_distance_mean_heavy"]
            d_rob = sym45.loc[cv, "cross_distance_robust_mean"] - sym135.loc[cv, "cross_distance_robust_mean"]
            d_cnt = sym45.loc[cv, "cross_contact_count_total"] - sym135.loc[cv, "cross_contact_count_total"]
            lines.append(f"| {int(cv)} | {e45:.0f} | {e135:.0f} | {pref} | {d_min:+.3f} | {d_rob:+.3f} | {d_cnt:+.0f} |")
        lines.append("")
        delta_e = [sym45.loc[cv, "relative_energy_meV"] - sym135.loc[cv, "relative_energy_meV"] for cv in common_c]
        corr_lines = []
        for col, label in crossover_descriptors() + [("cross_contact_count_total", "number of heavy-atom contacts < 4 Å (1-3 plus 2-4)")]:
            delta_d = [sym45.loc[cv, col] - sym135.loc[cv, col] for cv in common_c]
            r, n = pearson_r(delta_d, delta_e)
            corr_lines.append(f"  - {label}: Pearson r = {r:.2f} (n={n})")
        below = {cv: ("45/45" if sym45.loc[cv, "relative_energy_meV"] < sym135.loc[cv, "relative_energy_meV"] else "135/135") for cv in common_c if cv < 90}
        above = {cv: ("45/45" if sym45.loc[cv, "relative_energy_meV"] < sym135.loc[cv, "relative_energy_meV"] else "135/135") for cv in common_c if cv > 90}
        lines.append(
            f"For C<90° ({sorted(below)}) the lower-energy symmetric outer geometry is "
            f"{'consistently ' if len(set(below.values())) <= 1 else '(mixed) '}{sorted(set(below.values()))}; "
            f"for C>90° ({sorted(above)}) it is {'consistently ' if len(set(above.values())) <= 1 else '(mixed) '}{sorted(set(above.values()))}. "
            f"Correlation of ΔE(45−135) with the difference in cross-fragment packing:\n" + "\n".join(corr_lines) + "\n"
        )
        agree = {
            col: [((sym45.loc[cv, "relative_energy_meV"] < sym135.loc[cv, "relative_energy_meV"]) == (sym45.loc[cv, col] < sym135.loc[cv, col])) for cv in common_c]
            for col in ("cross_distance_mean_heavy", "cross_distance_robust_mean")
        }
        n_all = len(common_c)
        agree_txt = "; ".join(f"{col}: shorter-contact structure is the lower-energy one in {sum(v)}/{n_all} comparisons" for col, v in agree.items())
        lines.append(
            f"Consistency check ({agree_txt}). Cross-fragment packing therefore correlates with the handedness "
            "preference, but this is a correlation over a handful of symmetric comparisons, not proof of causation, and it "
            "does not mean the molecule generally minimizes d\u2081\u2083 or d\u2082\u2084: the closest distance alone does not "
            "explain the energy of arbitrary (L,R) geometries. The 3-shortest-contact mean and the contact count are "
            "included because a single closest atom pair is a fragile descriptor.\n"
        )
    else:
        lines.append("Not enough usable symmetric 45/45 and 135/135 points to test the crossover.\n")
    lines.append("![Crossover two-panel](plots/crossover_two_panel.png)\n")
    lines.append("Supplementary: ΔE vs Δd_cross scatter (Pearson r per descriptor).\n")
    lines.append("![Delta E vs delta cross contact](plots/delta_E_vs_delta_cross_contact.png)\n")

    # ---- Q5 ----
    lines.append("## Question 5 — Outer torsions → radical localization\n")
    outer_vals = sorted(set(df["L"].unique()) | set(df["R"].unique()))
    lines.append("| outer angle | frag1 mean±std (vs L) | frag1 range | frag4 mean±std (vs R) | frag4 range |")
    lines.append("|---|---|---|---|---|")
    for v in outer_vals:
        s1 = df.loc[usable & (df["L"] == v), "spin_fragment1_final"]
        s4 = df.loc[usable & (df["R"] == v), "spin_fragment4_final"]
        lines.append(f"| {int(v)} | {s1.mean():.3f}±{s1.std():.3f} | [{s1.min():.3f}, {s1.max():.3f}] | {s4.mean():.3f}±{s4.std():.3f} | [{s4.min():.3f}, {s4.max():.3f}] |")
    lines.append("")
    lines.append(
        "The scatter across all other coordinates is tiny (e.g. ~0.005 at 90°), so each terminal radical is controlled "
        "almost independently by its own outer torsion. Left and right behave symmetrically and are pooled in the main figure.\n"
    )
    lines.append("![Terminal spin vs outer torsion (pooled)](plots/terminal_spin_vs_outer_pooled.png)\n")
    lines.append("![Fragment spin vs outer torsion (L and R separately)](plots/fragment_spin_vs_outer_torsion.png)\n")
    lines.append("Control experiment (expected ~flat: varying the central torsion barely changes local radical character):\n")
    lines.append("![Control: terminal spin vs C at L=R=45](plots/terminal_spin_vs_C_control_L45R45.png)\n")
    r_s2, n_s2 = pearson_r(df.loc[usable, "mean_terminal_spin_final"], df.loc[usable, "s2_final"])
    lines.append(f"⟨S²⟩ vs mean terminal spin: Pearson r = {r_s2:.2f} (n={n_s2}).\n")
    lines.append("![S2 vs terminal spin localization](plots/s2_vs_terminal_spin_localization.png)\n")
    lines.append("![S2 vs C](plots/s2_vs_C.png)\n")
    lines.append("![Central backbone spin vs C](plots/central_backbone_spin_vs_C.png)\n")
    lines.append("![Radical atom 28/41 vs L (supplementary)](plots/radical_atom_spin_vs_L_supplementary.png)\n")

    # ---- Q6/Q7 ----
    lines.append("## Questions 6 & 7 — Radical localization → exchange coupling (PBE0 HS/BS)\n")
    if df["E_HS_pbe0"].notna().any():
        have = df["E_HS_pbe0"].notna() & usable
        r, n = pearson_r(df.loc[have, "mean_terminal_spin_final"], df.loc[have, "delta_E_exchange_meV"])
        n_triplet_ground = int((df.loc[have, "delta_E_exchange_meV"] < 0).sum())
        lines.append(
            f"PBE0 HS/BS single points found for {int(df['E_HS_pbe0'].notna().sum())} scan points. "
            f"Pearson r(mean terminal spin, ΔE_exchange) = {r:.2f} (n={n}). "
            f"Points where the raw HS energy is below BS (triplet locally favored at this fixed geometry): {n_triplet_ground}.\n"
        )
        lines.append("![Exchange vs C](plots/exchange_vs_C.png)\n")
        lines.append("![Exchange vs terminal spin](plots/exchange_vs_terminal_spin.png)\n")
    else:
        lines.append(
            "**No PBE0 HS/BS single-point data found yet.** The analysis picks up `<point>/high_spin/high_spin.out` and "
            "`<point>/broken_symmetry/broken_symmetry.out` automatically and will populate `delta_E_exchange_meV`, "
            "`s2_HS_pbe0`, `s2_BS_pbe0` and the fragment-1/4 HS/BS spins in `angle_scan.csv` and the exchange plots on the "
            "next run.\n"
        )
    lines.append(
        "**Limitations.** Every geometry here was optimized on the HIGH-SPIN R2SCAN-3C surface; a PBE0 HS/BS comparison on "
        "these frozen geometries gives the local electronic exchange at that HS-relaxed geometry and does not say whether "
        "the BS state would relax elsewhere. No D4 is needed on the PBE0 single points (it cancels in E_HS−E_BS at a "
        "shared geometry). The raw E_HS−E_BS is not a rigorous pure singlet-triplet gap.\n"
    )
    lines.append(
        "Once the exchange map exists, choose a small representative subset for independent BS geometry optimization "
        "(strongly AFM/high-conjugation, intermediate, nearly decoupled/localized, and any point where HS is competitive "
        "or lower) rather than repeating it over the full usable grid.\n"
    )

    # ---- Anomalies ----
    lines.append("## Anomalies\n")
    flagged = df[df["n_flags"] > 0]
    n_conv_anom = int(((df["status"] == "converged") & (df["n_flags"] > 0)).sum())
    lines.append(
        f"{len(flagged)} of {len(df)} points carry an anomaly flag: the {n_fail} non-converged jobs "
        f"(listed in `angle_scan.csv`, `status` column) plus {n_conv_anom} ORCA-converged points with a genuine flag, "
        "listed here. Routine convergence types, strained starting geometries, ring puckering and TRAH use are "
        "summarized above / in the `notes` column, not listed per point.\n"
    )
    genuine = df[(df["status"] == "converged") & (df["n_flags"] > 0)]
    if genuine.empty:
        lines.append("No ORCA-converged point has an anomaly flag.\n")
    else:
        for _, row in genuine.sort_values(["C", "L", "R"]).iterrows():
            lines.append(f"- **{row['label']}**: {row['flags']}")
        lines.append("")

    lines.append("## Per-C energy heatmaps over the (L, R) grid\n")
    for cv in sorted(df["C"].unique()):
        lines.append(f"![Energy heatmap C={int(cv)}](plots/heatmaps/energy_heatmap_C{int(cv):03d}.png)\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(df):
    n = len(df)
    print("=" * 70)
    print("Full-grid angle scan analysis summary")
    print("=" * 70)
    print(f"Scan points found       : {n}")
    print(f"  ORCA-converged        : {int((df['opt_converged']==True).sum())}")
    print(f"  chemically valid final: {int((df['final_geometry_valid']==True).sum())}")
    print(f"  usable                : {int(df['usable'].sum())}")
    usable = df["usable"]
    if usable.any():
        best = df.loc[usable, "relative_energy_meV"].idxmin()
        print(f"Global minimum          : {df.loc[best, 'label']} ({df.loc[best, 'final_energy']:.6f} Eh)")
    n_anom = int((df["n_flags"] > 0).sum())
    print(f"Points with anomalies   : {n_anom}/{n}")


if __name__ == "__main__":
    main()
