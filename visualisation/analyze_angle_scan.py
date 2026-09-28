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
    clash = min_dist < min_nonbonded_dist
    if clash:
        reasons.append(f"non-bonded clash: atoms {i}-{j} at {min_dist:.3f} Å")

    is_valid = (len(broken_bonds) == 0) and (not clash)
    return {
        "valid": is_valid,
        "bond_broken": len(broken_bonds) > 0,
        "clash": clash,
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
        if geometry_file.exists():
            try:
                g0, initial_atoms = analyze_geometry(geometry_file, CENTER, left_ring, right_ring)
                for k, v in g0.items():
                    row[f"init_{k}"] = v
                chk0 = check_chemical_integrity(initial_atoms, bonds, bonded_pairs, BOND_STRETCH_TOL_ANG, BOND_ABS_MAX_ANG, MIN_NONBONDED_DIST_ANG)
                row["initial_geometry_valid"] = chk0["valid"]
                row["initial_min_nonbonded_distance"] = chk0["min_nonbonded_distance"]
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
                row["bond_broken_during_optimization"] = chk1["bond_broken"] and row.get("initial_geometry_valid", True)
                row["clash_created_during_optimization"] = chk1["clash"] and row.get("initial_geometry_valid", True)

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

    # ---- CONSISTENCY FLAGS ----
    flags = {label: [] for label in df["label"]}
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
            flags[label].append(f"convergence type = {row['convergence_type']}: " + "; ".join(failing))
        if row.get("bond_broken_during_optimization") or row.get("clash_created_during_optimization"):
            flags[label].append(f"CHEMICAL BREAKDOWN DURING OPTIMIZATION: {row.get('chemical_break_reasons', '')}")
        elif row.get("initial_geometry_valid") is False and not row["usable"]:
            # Only worth surfacing when it actually explains a non-usable point --
            # a rigid pre-rotated guess can have a strained-but-harmless close
            # contact that relaxation trivially resolves, which is not an anomaly.
            flags[label].append("starting geometry already geometrically implausible (not a relaxation artifact)")
        if row.get("angle_maintained") is False:
            flags[label].append("constrained angle not maintained within tolerance")
        s2f = row.get("s2_final")
        if s2f is not None and pd.notna(s2f) and abs(s2f - IDEAL_S2_TRIPLET) > S2_ANOMALY_THRESHOLD:
            flags[label].append(f"final <S^2>={s2f:.3f} anomalous for a triplet")
        for ring_name in ("left", "right"):
            rms = row.get(f"final_{ring_name}_plane_rms")
            if rms is not None and pd.notna(rms) and rms > PLANARITY_RMS_THRESHOLD_ANG:
                flags[label].append(f"{ring_name} ring non-planar (RMS={rms:.3f} Å)")
        if row.get("scf_not_converged"):
            flags[label].append("SCF NOT CONVERGED reported")
        if row.get("trah_activations", 0) > 0:
            flags[label].append(f"TRAH activated ({row['trah_activations']} times)")
        if row.get("neg_hessian_events", 0) > 0:
            flags[label].append(f"negative Hessian eigenvalues ({row['neg_hessian_events']} events)")

    # Electronic/branch-change anomaly: energy discontinuity vs a grid
    # neighbor is only flagged as a possible ELECTRONIC anomaly if it
    # coincides with a discontinuity in <S^2> or terminal spin along the SAME
    # neighbor pair. A large energy change vs C alone is normal steric/
    # conjugation physics and is NOT flagged by itself (see angle_scan.md).
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
                if e_jump and (s2_jump or spin_jump):
                    flags[label].append(
                        f"possible electronic/SCF branch change vs {nrow['label']} "
                        f"(ΔE={row['relative_energy_meV']-nrow['relative_energy_meV']:.0f} meV, "
                        f"{'Δ<S^2>' if s2_jump else ''}{' & ' if s2_jump and spin_jump else ''}{'Δspin' if spin_jump else ''} also jumps)"
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
    make_plots(df, PLOTS_DIR, HEATMAPS_DIR, RADICAL_ATOMS)
    print(f"Wrote plots to {PLOTS_DIR}")

    # ---- MARKDOWN REPORT ----
    write_markdown_report(df, MD_PATH, CENTER, LEFT, RIGHT, left_ring, right_ring, fragment1, fragment4, ROOT_DIR, ref_row)
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


def make_plots(df, plots_dir, heatmaps_dir, radical_atoms):
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

    # --- PBE0 HS/BS plots (only if data present) ---
    if df["E_HS_pbe0"].notna().any():
        make_pbe0_plots(df, plots_dir)


def symmetric_subset(df, l_val, r_val):
    return df[(df["L"] == l_val) & (df["R"] == r_val) & df["usable"]].set_index("C")


def make_crossover_plots(df, plots_dir, c_values):
    sym45 = symmetric_subset(df, 45.0, 45.0)
    sym135 = symmetric_subset(df, 135.0, 135.0)
    common_c = sorted(set(sym45.index) & set(sym135.index))
    if not common_c:
        return

    delta_e = [sym45.loc[cv, "relative_energy_meV"] - sym135.loc[cv, "relative_energy_meV"] for cv in common_c]
    delta_cross = [sym45.loc[cv, "cross_distance_mean_heavy"] - sym135.loc[cv, "cross_distance_mean_heavy"] for cv in common_c]

    fig, ax = plt.subplots()
    ax.plot(common_c, [sym45.loc[cv, "min_distance_1_3_heavy"] for cv in common_c], "o-", label="dmin(1,3), 45/45")
    ax.plot(common_c, [sym45.loc[cv, "min_distance_2_4_heavy"] for cv in common_c], "s-", label="dmin(2,4), 45/45")
    ax.plot(common_c, [sym135.loc[cv, "min_distance_1_3_heavy"] for cv in common_c], "o--", label="dmin(1,3), 135/135")
    ax.plot(common_c, [sym135.loc[cv, "min_distance_2_4_heavy"] for cv in common_c], "s--", label="dmin(2,4), 135/135")
    ax.set_xlabel("Central (C) target dihedral (deg)")
    ax.set_ylabel("Cross-fragment min. heavy-atom distance (Å)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "cross_contacts_45_vs_135.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.axhline(0, color="k", linewidth=1)
    ax.plot(common_c, delta_e, "o-", color="tab:purple")
    ax.set_xlabel("Central (C) target dihedral (deg)")
    ax.set_ylabel("ΔE = E(45/45) − E(135/135) (meV)")
    ax.set_title("Negative = 45/45 preferred, positive = 135/135 preferred")
    fig.tight_layout()
    fig.savefig(plots_dir / "delta_E_45_vs_135.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(delta_cross, delta_e)
    r, n = pearson_r(delta_cross, delta_e)
    for cv, x, y in zip(common_c, delta_cross, delta_e):
        ax.annotate(f"{int(cv)}", (x, y), fontsize=7)
    ax.set_xlabel("Δ cross-contact distance, 45/45 − 135/135 (Å)")
    ax.set_ylabel("ΔE, 45/45 − 135/135 (meV)")
    ax.set_title(f"Pearson r = {r:.2f} (n={n}) — correlation, not causation")
    fig.tight_layout()
    fig.savefig(plots_dir / "delta_E_vs_delta_cross_contact.png", dpi=150)
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


def write_markdown_report(df, md_path, center, left, right, left_ring, right_ring, fragment1, fragment4, root_dir, ref_row):
    lines = []
    lines.append("# ORCA constrained-torsion angle scan — full L×C×R grid\n")
    lines.append("## Overview and methods\n")
    lines.append(f"- Scan root: `{root_dir}`")
    lines.append(f"- {len(df)} scan points, R2SCAN-3C TightOpt UKS, high-spin (multiplicity 3)")
    lines.append(f"- CENTER dihedral {center} (bond {center[1]}-{center[2]}), LEFT {left}, RIGHT {right}")
    lines.append(f"- Fragment 1 (LEFT terminal, radical atom 41, {len(fragment1)} atoms): {fragment1}")
    lines.append(f"- Fragment 4 (RIGHT terminal, radical atom 28, {len(fragment4)} atoms): {fragment4}")
    lines.append("- Fragments 2/3: the central backbone, split at the CENTER bond")
    lines.append(
        "- **Convergence** is read directly from ORCA's literal `THE OPTIMIZATION HAS CONVERGED` banner "
        "(authoritative), with the override type it used (strict / gradient-overachieved / "
        "step-overachieved / energy-override / other) recorded separately for transparency."
    )
    lines.append(
        "- **Chemical integrity** is checked on BOTH the initial and final geometry against the clean, "
        "untwisted baseline connectivity (bond stretch > 0.30 Å or length > 1.9 Å = broken bond; "
        "non-bonded pair < 1.5 Å = clash). `bond_broken_during_optimization` / "
        "`clash_created_during_optimization` are only true if the INITIAL geometry was already valid — "
        "i.e. the breakdown happened during relaxation, not because the starting guess was already "
        "geometrically impossible."
    )
    lines.append(
        "- **Spin populations**: `spin_final`/`s2_final` require BOTH ORCA convergence (the banner above) "
        "AND a post-optimization Mulliken population analysis. Unconverged jobs never contribute to "
        "primary plots; their last available (initial-SCF) spin is kept separately as "
        "`radical_spin_last_available_*` for bookkeeping only."
    )
    lines.append(
        "- **Energy discontinuity vs. a neighboring grid point is NOT by itself flagged as anomalous** "
        "(large curvature near C~20°/140-160° is expected steric/conjugation physics). It is only "
        "flagged as a *possible electronic/SCF branch change* when it coincides with a simultaneous jump "
        "in ⟨S²⟩ or terminal-fragment spin at the same neighbor pair."
    )
    lines.append(
        "- **Magnitude/handedness**: for each of L, C, R, `magnitude = min(θ, 180−θ)` and "
        "`handedness = sign(θ−90)`. 45° and 135° share the same magnitude but opposite "
        "handedness and are NOT treated as equivalent anywhere in this analysis."
    )
    lines.append("")

    usable = df["usable"]

    # ---- Q1: accessible space ----
    lines.append("## Question 1 — What L,C,R space is mechanically/chemically accessible?\n")
    lines.append("| C (deg) | ORCA-converged / 25 | geometry valid (final) | usable |")
    lines.append("|---|---|---|---|")
    for cv in sorted(df["C"].unique()):
        sub = df[df["C"] == cv]
        lines.append(f"| {int(cv)} | {(sub['status']=='converged').sum()}/{len(sub)} | {(sub['final_geometry_valid']==True).sum()} | {sub['usable'].sum()} |")
    lines.append("")
    lines.append("![Convergence vs C](plots/convergence_fraction_vs_C.png)\n")
    lines.append("![Optimization cycles vs C](plots/optimization_cycles_vs_C.png)\n")
    lines.append("![Steric vs C](plots/steric_vs_C.png)\n")

    broken_converged = df[(df["status"] == "converged") & (df["final_geometry_valid"] == False)]
    lines.append(f"**Converged-but-chemically-broken points** (ORCA's own convergence signal does not catch these): {len(broken_converged)}\n")
    for _, row in broken_converged.iterrows():
        lines.append(f"- **{row['label']}**: E={row['final_energy']:.6f} Eh; {row['chemical_break_reasons']}")
    lines.append("")

    # ---- Q2: central torsion accommodation ----
    lines.append("## Question 2 — How does the molecule accommodate the imposed central torsion?\n")
    lines.append(
        "ORCA holds the literal 4-atom CENTER dihedral exactly at target; the fitted central-plane angle "
        "(between the two aromatic rings flanking the CENTER bond) can still differ, absorbing torsion "
        "through local puckering/bond tilt instead.\n"
    )
    lines.append("![Plane angle vs target](plots/plane_angle_vs_target.png)\n")
    lines.append("![Mean plane angle vs C](plots/mean_plane_angle_vs_C.png)\n")
    lines.append("![Plane angle deviation vs C](plots/plane_angle_deviation_vs_C.png)\n")
    lines.append("![Bond tilt vs C](plots/bond_tilt_vs_C.png)\n")
    lines.append("![Bending accommodation vs C](plots/bending_accommodation_vs_C.png)\n")

    # ---- Q3/Q4: crossover + cross contacts ----
    lines.append("## Questions 3 & 4 — The 45/45 ↔ 135/135 outer-folding crossover\n")
    sym45 = symmetric_subset(df, 45.0, 45.0)
    sym135 = symmetric_subset(df, 135.0, 135.0)
    common_c = sorted(set(sym45.index) & set(sym135.index))
    if common_c:
        lines.append("| C | E(45/45) rel. (meV) | E(135/135) rel. (meV) | preferred | Δcross-contact (Å), 45−135 |")
        lines.append("|---|---|---|---|---|")
        for cv in common_c:
            e45, e135 = sym45.loc[cv, "relative_energy_meV"], sym135.loc[cv, "relative_energy_meV"]
            pref = "45/45" if e45 < e135 else "135/135"
            dcross = sym45.loc[cv, "cross_distance_mean_heavy"] - sym135.loc[cv, "cross_distance_mean_heavy"]
            lines.append(f"| {int(cv)} | {e45:.0f} | {e135:.0f} | {pref} | {dcross:+.3f} |")
        lines.append("")
        delta_e = [sym45.loc[cv, "relative_energy_meV"] - sym135.loc[cv, "relative_energy_meV"] for cv in common_c]
        delta_cross = [sym45.loc[cv, "cross_distance_mean_heavy"] - sym135.loc[cv, "cross_distance_mean_heavy"] for cv in common_c]
        r, n = pearson_r(delta_cross, delta_e)
        below90_pref = {cv: ("45/45" if sym45.loc[cv, "relative_energy_meV"] < sym135.loc[cv, "relative_energy_meV"] else "135/135") for cv in common_c if cv < 90}
        above90_pref = {cv: ("45/45" if sym45.loc[cv, "relative_energy_meV"] < sym135.loc[cv, "relative_energy_meV"] else "135/135") for cv in common_c if cv > 90}
        below_consistent = len(set(below90_pref.values())) <= 1
        above_consistent = len(set(above90_pref.values())) <= 1
        lines.append(
            f"Crossover check: for C<90° ({sorted(below90_pref)} deg), the preferred structure is "
            f"{'consistently ' if below_consistent else '(mixed) '}{list(below90_pref.values())}; "
            f"for C>90° ({sorted(above90_pref)} deg), it is "
            f"{'consistently ' if above_consistent else '(mixed) '}{list(above90_pref.values())}. "
            f"Correlation between ΔE(45−135) and Δcross-contact "
            f"distance: **Pearson r = {r:.2f}** (n={n}). Correlation does not establish causation — see "
            f"`delta_E_vs_delta_cross_contact.png` and cross-check against plane angle / bond tilt / "
            f"puckering / bond-angle strain in `angle_scan.csv` before drawing mechanistic conclusions.\n"
        )
    else:
        lines.append("Not enough usable symmetric 45/45 and 135/135 points to test the crossover.\n")
    lines.append("![Cross contacts 45 vs 135](plots/cross_contacts_45_vs_135.png)\n")
    lines.append("![Delta E 45 vs 135](plots/delta_E_45_vs_135.png)\n")
    lines.append("![Delta E vs delta cross contact](plots/delta_E_vs_delta_cross_contact.png)\n")

    # ---- Q5: independence of outer torsions ----
    lines.append("## Question 5 — Do L and R independently control fragment 1 and 4 spin?\n")
    outer_vals = sorted(set(df["L"].unique()) | set(df["R"].unique()))
    lines.append("| outer angle | frag1 mean±std (vs L) | frag1 range | frag4 mean±std (vs R) | frag4 range |")
    lines.append("|---|---|---|---|---|")
    for v in outer_vals:
        s1 = df.loc[usable & (df["L"] == v), "spin_fragment1_final"]
        s4 = df.loc[usable & (df["R"] == v), "spin_fragment4_final"]
        lines.append(
            f"| {int(v)} | {s1.mean():.3f}±{s1.std():.3f} | [{s1.min():.3f}, {s1.max():.3f}] | "
            f"{s4.mean():.3f}±{s4.std():.3f} | [{s4.min():.3f}, {s4.max():.3f}] |"
        )
    lines.append("")
    lines.append("![Fragment spin vs outer torsion](plots/fragment_spin_vs_outer_torsion.png)\n")
    lines.append("![Central backbone spin vs C](plots/central_backbone_spin_vs_C.png)\n")
    lines.append("![Radical atom 28/41 vs L (supplementary)](plots/radical_atom_spin_vs_L_supplementary.png)\n")
    lines.append("![S2 vs C](plots/s2_vs_C.png)\n")
    lines.append("![S2 vs terminal spin localization](plots/s2_vs_terminal_spin_localization.png)\n")

    # ---- Q6/Q7: exchange (scaffold) ----
    lines.append("## Questions 6 & 7 — Does terminal localization suppress the AFM HS−BS exchange?\n")
    if df["E_HS_pbe0"].notna().any():
        have = df["E_HS_pbe0"].notna() & usable
        r, n = pearson_r(df.loc[have, "mean_terminal_spin_final"], df.loc[have, "delta_E_exchange_meV"])
        n_triplet_ground = int((df.loc[have, "delta_E_exchange_meV"] < 0).sum())
        lines.append(
            f"PBE0 HS/BS single points found for {int(df['E_HS_pbe0'].notna().sum())} scan points. "
            f"Pearson r(mean terminal spin, ΔE_exchange) = {r:.2f} (n={n}). "
            f"Points where the raw HS energy is BELOW BS (triplet locally favored at this fixed geometry): {n_triplet_ground}.\n"
        )
        lines.append("![Exchange vs C](plots/exchange_vs_C.png)\n")
        lines.append("![Exchange vs terminal spin](plots/exchange_vs_terminal_spin.png)\n")
    else:
        lines.append(
            "**No PBE0 HS/BS single-point data found yet.** This analysis is wired up to automatically pick "
            "up `<point>/high_spin/high_spin.out` and `<point>/broken_symmetry/broken_symmetry.out` for each "
            "scan point once those separate PBE0 single points exist, and will populate "
            "`delta_E_exchange_meV`, `s2_HS_pbe0`, `s2_BS_pbe0`, and the fragment-1/4 HS/BS spins in "
            "`angle_scan.csv` plus the plots above on the next run — no code changes needed.\n"
        )
    lines.append(
        "**Important limitation**: every geometry here was optimized on the HIGH-SPIN R2SCAN-3C surface. "
        "A PBE0 HS/BS comparison on these frozen geometries gives the local electronic exchange AT that "
        "high-spin-relaxed geometry — it does NOT determine whether the broken-symmetry state would relax "
        "to a different geometry. D4 dispersion should not be added to the PBE0 HS/BS single points: both "
        "states share the same frozen geometry, so it cancels in the difference; the raw E_HS−E_BS is "
        "also not a rigorous pure singlet-triplet gap and is labeled as such throughout.\n"
    )
    lines.append(
        "Once the exchange map is populated, a small representative subset (strongly AFM point, "
        "intermediate point, near-decoupled/localized point, and any point where HS becomes competitive "
        "with or lower than BS) should be selected for independent BS geometry optimization, rather than "
        "repeating it over the full usable grid.\n"
    )

    lines.append("## Flagged anomalies\n")
    flagged = df[df["n_flags"] > 0]
    if flagged.empty:
        lines.append("None.\n")
    else:
        for _, row in flagged.sort_values(["C", "L", "R"]).iterrows():
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
