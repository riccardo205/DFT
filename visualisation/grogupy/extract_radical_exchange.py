#!/usr/bin/env python3
"""
Post-process delocalized radical sites from a SIESTA non-collinear calculation
and extract exchange tensors with grogupy.

This script intentionally separates two things:

1.  A SIESTA *.out file contains atomic spin populations.  These are useful to
    define and validate *magnetic entities*, but they do NOT contain exchange
    constants by themselves.

2.  Exchange constants are curvatures of the electronic energy with respect to
    infinitesimal local rotations.  The `--run-grogupy` mode therefore reads
    the SIESTA .fdf plus its saved Hamiltonian / density-matrix sidecars and
    applies grogupy's Green-function / torque workflow.

The atom groups file is assumed to use SIESTA's 1-based atom indexing.  grogupy
(and sisl) use 0-based indices, so this conversion is performed only inside
`run_grogupy`.

Recommended usage for the present four-radical dimer
----------------------------------------------------
# A. Validate the final SIESTA spin texture:
python extract_radical_exchange.py \
  --out siesta.out \
  --groups conf_11_antiferro1_Sz_groups86_separated_atoms.txt \
  --report radical_spin_report.json

# B. Run the torque extraction in the directory that contains the .fdf and
#    the files emitted by SIESTA (e.g. .TSHS and DM / DMHS NetCDF files):
python extract_radical_exchange.py \
  --fdf dimers-short-2OS.fdf \
  --groups conf_11_antiferro1_Sz_groups86_separated_atoms.txt \
  --run-grogupy --kmesh 1 1 1 --eset 400 --esetp 10000 \
  --assume-no-soc \
  --magnopy radical_4site_allPAO.magnopy \
  --exchange-json radical_4site_allPAO.json

# C. Repeat B with `--orbital-model p` and compare the two JSON files.
#    Do not quote a J until the result is stable against this projector choice
#    and against a denser contour (e.g. --eset 800).

Dependencies:
  - standard report mode: Python >=3.9, numpy
  - torque mode: grogupy==0.3.1, sisl, numpy

Conventions:
  grogupy writes the normalized-spin Hamiltonian
      H = 1/2 sum_{i != j} e_i^T J_ij e_j + sum_i e_i^T K_i e_i,
  where e_i are unit vectors.  Consequently its J values are the DFT-matched
  energies for *unit* local moments.  With this convention J_iso > 0 favors
  antiparallel alignment and J_iso < 0 favors parallel alignment.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


# Fixed atom ordering in the supplied SIESTA structure:
# molecule 1: C 1--54, H 55--86
# molecule 2: C 87--140, H 141--172
DEFAULT_EXPECTED_CARBON = list(range(1, 55)) + list(range(87, 141))
ENTITY_LABELS = ("A_mol1_plus", "B_mol1_minus", "C_mol2_plus", "D_mol2_minus")
PAIR_SPECS = (
    (0, 1, "A-B  intramolecular, molecule 1"),
    (2, 3, "C-D  intramolecular, molecule 2"),
    (0, 2, "A-C  intermolecular, parallel sector"),
    (0, 3, "A-D  intermolecular, crossed sector"),
    (1, 2, "B-C  intermolecular, crossed sector"),
    (1, 3, "B-D  intermolecular, parallel sector"),
)


@dataclass(frozen=True)
class AtomicSpin:
    """One row from a SIESTA Mulliken atomic-population table."""

    atom: int  # SIESTA 1-based atom index
    charge: float
    valence: float
    spin_abs: float
    sx: float
    sy: float
    sz: float
    species: str

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.sx, self.sy, self.sz], dtype=float)


def _as_jsonable(value: Any) -> Any:
    """Convert NumPy / Path objects recursively to JSON-friendly values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    return value


def parse_groups(path: str | Path) -> dict[int, list[int]]:
    """
    Parse the supplied 'GROUP n ... [atom ids]' text file.

    The routine deliberately accepts line breaks inside the [] list and rejects
    duplicated group labels or empty groups.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"GROUP\s+(\d+).*?\n\s*\[(.*?)\]\s*\n\s*Count",
        flags=re.IGNORECASE | re.DOTALL,
    )
    parsed: dict[int, list[int]] = {}
    for match in pattern.finditer(text):
        group_id = int(match.group(1))
        atoms = [int(token) for token in re.findall(r"\d+", match.group(2))]
        if group_id in parsed:
            raise ValueError(f"Group {group_id} appears more than once in {path}.")
        if not atoms:
            raise ValueError(f"Group {group_id} is empty in {path}.")
        parsed[group_id] = atoms

    if not parsed:
        raise ValueError(
            f"Could not find blocks like 'GROUP n ... [1, 2, ...]' in {path}."
        )
    return dict(sorted(parsed.items()))


_MULLIKEN_HEADER = re.compile(r"^\s*Mulliken Atomic Populations:\s*$", re.I)
_MULLIKEN_ROW = re.compile(
    r"^\s*(\d+)\s+"
    r"([-+0-9.EeDd]+)\s+"  # charge
    r"([-+0-9.EeDd]+)\s+"  # valence
    r"([-+0-9.EeDd]+)\s+"  # |S|
    r"([-+0-9.EeDd]+)\s+"  # Sx
    r"([-+0-9.EeDd]+)\s+"  # Sy
    r"([-+0-9.EeDd]+)\s+"  # Sz
    r"(\S+)\s*$"            # species
)


def _to_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def parse_final_mulliken_table(path: str | Path) -> dict[int, AtomicSpin]:
    """
    Read the last complete Mulliken atomic-population table in a SIESTA .out.

    SIESTA may print the same table more than once.  The last complete table is
    usually the final result, which is what we want for entity validation.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    starts = [idx for idx, line in enumerate(lines) if _MULLIKEN_HEADER.match(line)]
    if not starts:
        raise ValueError(f"No Mulliken Atomic Populations table found in {path}.")

    candidates: list[dict[int, AtomicSpin]] = []
    for start in starts:
        table: dict[int, AtomicSpin] = {}
        for line in lines[start + 1 :]:
            match = _MULLIKEN_ROW.match(line)
            if match is not None:
                values = match.groups()
                atom = int(values[0])
                table[atom] = AtomicSpin(
                    atom=atom,
                    charge=_to_float(values[1]),
                    valence=_to_float(values[2]),
                    spin_abs=_to_float(values[3]),
                    sx=_to_float(values[4]),
                    sy=_to_float(values[5]),
                    sz=_to_float(values[6]),
                    species=values[7],
                )
                continue

            # Once at least one row has been collected, the next unrelated line
            # marks the end of this table.
            if table:
                break

        if table:
            candidates.append(table)

    if not candidates:
        raise ValueError(f"Found Mulliken headers but no readable rows in {path}.")

    final = candidates[-1]
    atom_ids = sorted(final)
    expected = list(range(1, atom_ids[-1] + 1))
    if atom_ids != expected:
        raise ValueError(
            "The selected Mulliken table is incomplete or atom numbering is not "
            f"continuous. Read atoms {atom_ids[:3]} ... {atom_ids[-3:]}."
        )
    return final


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = _norm(a), _norm(b)
    if na < 1e-14 or nb < 1e-14:
        return None
    cosine = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def validate_groups(
    groups: dict[int, list[int]],
    atom_spins: dict[int, AtomicSpin] | None = None,
    expected_carbon: Sequence[int] = DEFAULT_EXPECTED_CARBON,
) -> dict[str, Any]:
    """
    Check that magnetic entities are disjoint and cover the intended C network.

    A partition test is essential here: placing one PAO in two entities makes
    the fitted pair interactions ambiguous, while omitting part of the radical
    support makes the local moment artificially small.
    """
    group_ids = list(groups)
    flattened = [atom for gid in group_ids for atom in groups[gid]]
    unique = set(flattened)
    duplicate_atoms = sorted({x for x in flattened if flattened.count(x) > 1})
    expected = set(expected_carbon)
    missing = sorted(expected - unique)
    extra = sorted(unique - expected)

    report: dict[str, Any] = {
        "group_ids": group_ids,
        "n_groups": len(group_ids),
        "n_atoms_across_groups": len(flattened),
        "n_unique_atoms": len(unique),
        "duplicate_atoms": duplicate_atoms,
        "expected_carbon_count": len(expected),
        "missing_expected_carbons": missing,
        "extra_atoms_outside_expected_carbons": extra,
        "partition_ok": not duplicate_atoms and not missing and not extra,
    }

    if atom_spins is not None:
        unavailable = sorted(unique - set(atom_spins))
        report["atoms_missing_from_mulliken_table"] = unavailable
        if unavailable:
            report["partition_ok"] = False
    return report


def spin_report(
    atom_spins: dict[int, AtomicSpin],
    groups: dict[int, list[int]],
    expected_carbon: Sequence[int] = DEFAULT_EXPECTED_CARBON,
) -> dict[str, Any]:
    """Return local spin vectors, coherence, residual spin, and inter-group angles."""
    validation = validate_groups(groups, atom_spins, expected_carbon)
    entity_records: list[dict[str, Any]] = []
    vectors: dict[int, np.ndarray] = {}

    for label, (gid, atoms) in zip(ENTITY_LABELS, groups.items(), strict=False):
        spin_vectors = np.array([atom_spins[i].vector for i in atoms], dtype=float)
        net = spin_vectors.sum(axis=0)
        scalar_sum = float(sum(atom_spins[i].spin_abs for i in atoms))
        net_norm = _norm(net)
        coherence = None if scalar_sum < 1e-14 else net_norm / scalar_sum
        vectors[gid] = net
        entity_records.append(
            {
                "label": label,
                "group": gid,
                "n_atoms": len(atoms),
                "atoms_siesta_1based": atoms,
                "spin_vector_muB": net,
                "spin_magnitude_muB": net_norm,
                "sum_atomic_magnitudes_muB": scalar_sum,
                # 1 means all atomic moments are parallel inside the entity.
                "intra_entity_collinearity": coherence,
                "mean_charge_e": float(np.mean([atom_spins[i].charge for i in atoms])),
            }
        )

    all_group_atoms = set(i for atoms in groups.values() for i in atoms)
    residual_carbon = set(expected_carbon) - all_group_atoms
    residual_vector = sum(
        (atom_spins[i].vector for i in residual_carbon),
        start=np.zeros(3, dtype=float),
    )
    all_atoms = sorted(atom_spins)
    total_vector = sum(
        (atom_spins[i].vector for i in all_atoms),
        start=np.zeros(3, dtype=float),
    )

    pair_angles: list[dict[str, Any]] = []
    for g1, g2 in combinations(groups, 2):
        pair_angles.append(
            {
                "group_i": g1,
                "group_j": g2,
                "angle_deg": _angle_deg(vectors[g1], vectors[g2]),
            }
        )

    return {
        "group_validation": validation,
        "entities": entity_records,
        "angles_between_entity_net_spins": pair_angles,
        "residual_expected_carbon_spin_muB": residual_vector,
        "residual_expected_carbon_magnitude_muB": _norm(residual_vector),
        "total_cell_spin_muB": total_vector,
        "total_cell_spin_magnitude_muB": _norm(total_vector),
    }


def print_spin_report(report: dict[str, Any]) -> None:
    """Human-readable terminal summary."""
    val = report["group_validation"]
    print("\n=== entity partition ===")
    print(
        f"{val['n_groups']} groups, {val['n_unique_atoms']} unique selected atoms; "
        f"partition_ok={val['partition_ok']}"
    )
    if val["duplicate_atoms"]:
        print(f"  duplicated atoms: {val['duplicate_atoms']}")
    if val["missing_expected_carbons"]:
        print(f"  missing expected C atoms: {val['missing_expected_carbons']}")
    if val["extra_atoms_outside_expected_carbons"]:
        print(f"  unexpected selected atoms: {val['extra_atoms_outside_expected_carbons']}")

    print("\n=== net moment of each radical entity (Mulliken, mu_B) ===")
    for entity in report["entities"]:
        vx, vy, vz = entity["spin_vector_muB"]
        print(
            f"{entity['label']:24s}  "
            f"M=({vx:+.6f}, {vy:+.6f}, {vz:+.6f})  "
            f"|M|={entity['spin_magnitude_muB']:.6f}  "
            f"coherence={entity['intra_entity_collinearity']:.6f}"
        )

    print("\n=== angles between entity net moments ===")
    for pair in report["angles_between_entity_net_spins"]:
        angle = pair["angle_deg"]
        printable = "undefined" if angle is None else f"{angle:.3f} deg"
        print(f"group {pair['group_i']} -- group {pair['group_j']}: {printable}")

    total = report["total_cell_spin_muB"]
    print(
        "\nTotal cell spin: "
        f"({total[0]:+.6f}, {total[1]:+.6f}, {total[2]:+.6f}) mu_B; "
        f"|M|={report['total_cell_spin_magnitude_muB']:.6f} mu_B"
    )


def decompose_exchange_tensor(J_meV: np.ndarray) -> dict[str, Any]:
    """
    Split a real 3x3 exchange tensor into isotropic, traceless symmetric,
    and antisymmetric/DM pieces.

    The DM convention is:
      D = ((J_yz-J_zy)/2, (J_zx-J_xz)/2, (J_xy-J_yx)/2).
    """
    J = np.asarray(J_meV, dtype=float)
    if J.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 exchange tensor, got {J.shape}.")
    J_iso = float(np.trace(J) / 3.0)
    symmetric = 0.5 * (J + J.T)
    symmetric_traceless = symmetric - J_iso * np.eye(3)
    antisymmetric = 0.5 * (J - J.T)
    D = np.array(
        [antisymmetric[1, 2], antisymmetric[2, 0], antisymmetric[0, 1]],
        dtype=float,
    )
    scale = max(abs(J_iso), np.linalg.norm(J) / math.sqrt(3), 1e-12)
    return {
        "J_meV": J,
        "J_iso_meV": J_iso,
        "J_symmetric_traceless_meV": symmetric_traceless,
        "D_meV": D,
        "norm_symmetric_traceless_meV": float(np.linalg.norm(symmetric_traceless)),
        "norm_D_meV": float(np.linalg.norm(D)),
        "relative_symmetric_anisotropy": float(np.linalg.norm(symmetric_traceless) / scale),
        "relative_DM": float(np.linalg.norm(D) / scale),
    }


def no_soc_tensor_check(J_meV: np.ndarray, tolerance: float = 0.05) -> dict[str, Any]:
    """
    A physics check for a non-SOC calculation.

    Without spin-orbit coupling, a correctly mapped isolated spin system must
    be invariant under a common spin rotation.  Therefore, exchange should be
    isotropic (J proportional to I) and the DM part should vanish within
    numerical integration/projector error.
    """
    d = decompose_exchange_tensor(J_meV)
    return {
        **d,
        "tolerance": float(tolerance),
        "passes_isotropy_check": (
            d["relative_symmetric_anisotropy"] <= tolerance
            and d["relative_DM"] <= tolerance
        ),
    }


def _entity_specs(groups: dict[int, list[int]], orbital_model: str) -> list[dict[str, Any]]:
    """
    Construct grogupy cluster selections.

    SIESTA IDs are 1-based; sisl / grogupy atom indices are 0-based.
    """
    specs: list[dict[str, Any]] = []
    for atoms_1based in groups.values():
        atoms_0based = [a - 1 for a in atoms_1based]
        spec: dict[str, Any] = {"atom": atoms_0based}
        if orbital_model == "p":
            # One l=1 shell definition for each carbon atom in the cluster.
            spec["l"] = [[1] for _ in atoms_0based]
        specs.append(spec)
    return specs


def _extract_grogupy_results(simulation: Any, assume_no_soc: bool) -> dict[str, Any]:
    """Collect documented grogupy result attributes into a compact JSON record."""
    entities_out: list[dict[str, Any]] = []
    for label, entity in zip(ENTITY_LABELS, simulation.magnetic_entities, strict=False):
        record: dict[str, Any] = {
            "label": label,
            "tag": getattr(entity, "tag", None),
            "local_charge_e": getattr(entity, "local_Q", None),
            "local_spin_magnitude_muB": getattr(entity, "local_S", None),
            "local_spin_vector_muB": [
                getattr(entity, "local_Sx", None),
                getattr(entity, "local_Sy", None),
                getattr(entity, "local_Sz", None),
            ],
            "total_spin_magnitude_muB": getattr(entity, "total_S", None),
            "total_spin_vector_muB": [
                getattr(entity, "total_Sx", None),
                getattr(entity, "total_Sy", None),
                getattr(entity, "total_Sz", None),
            ],
            "K_meV": getattr(entity, "K_meV", None),
            "K_consistency_meV": getattr(entity, "K_consistency_meV", None),
            "xyz_center_Ang": getattr(entity, "xyz_center", None),
        }
        entities_out.append(record)

    pairs_out: list[dict[str, Any]] = []
    for (_, _, label), pair in zip(PAIR_SPECS, simulation.pairs, strict=False):
        J = np.asarray(pair.J_meV, dtype=float)
        d = decompose_exchange_tensor(J)
        record = {
            "label": label,
            "entity_tags": getattr(pair, "tags", None),
            "distance_Ang": getattr(pair, "distance", None),
            "J_meV": J,
            "J_iso_meV": d["J_iso_meV"],
            "J_symmetric_traceless_meV": d["J_symmetric_traceless_meV"],
            "D_meV": d["D_meV"],
            "relative_symmetric_anisotropy": d["relative_symmetric_anisotropy"],
            "relative_DM": d["relative_DM"],
        }
        if assume_no_soc:
            record["no_SOC_check"] = no_soc_tensor_check(J)
        pairs_out.append(record)

    return {
        "magnetic_entities": entities_out,
        "pairs": pairs_out,
        "grogupy_magnopy_text": simulation.to_magnopy(comments=False),
    }


def run_grogupy(
    fdf_path: str | Path,
    groups: dict[int, list[int]],
    *,
    kmesh: Sequence[int],
    emin: float | None,
    eset: int,
    esetp: int,
    scf_axis: Sequence[float],
    orbital_model: str,
    spin_model: str,
    assume_no_soc: bool,
    magnopy_out: str | Path,
    exchange_json: str | Path,
) -> dict[str, Any]:
    """
    Run grogupy's torque/Green-function extraction for the four cluster entities.

    The `.fdf` must be in the original SIESTA run directory, alongside the
    Hamiltonian and density-matrix sidecars named by SystemLabel.
    """
    try:
        import grogupy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "grogupy is required only for --run-grogupy. Install it with:\n"
            "  python -m pip install 'grogupy==0.3.1'\n"
        ) from exc

    fdf_path = Path(fdf_path)
    if not fdf_path.exists():
        raise FileNotFoundError(f"No .fdf file found at {fdf_path}")

    if len(groups) != 4:
        raise ValueError(
            f"This workflow is configured for four radical entities; found {len(groups)} groups."
        )

    if emin is None:
        try:
            # In grogupy 0.3.1 this helper lives in the physics utilities
            # module, rather than the package top-level namespace.
            from grogupy.physics.utilities import automatic_emin
        except ImportError as exc:
            raise RuntimeError(
                "Could not import grogupy.physics.utilities.automatic_emin. "
                "Supply --emin manually, below the lowest SIESTA Hamiltonian eigenvalue."
            ) from exc
        emin = float(automatic_emin(str(fdf_path)))

    kmesh = [int(v) for v in kmesh]
    if len(kmesh) != 3 or any(v <= 0 for v in kmesh):
        raise ValueError("--kmesh must contain three positive integers, e.g. 1 1 1.")

    hamiltonian = grogupy.Hamiltonian(
        infile=str(fdf_path),
        scf_xcf_orientation=np.asarray(scf_axis, dtype=float),
    )
    # grogupy 0.3.1 starts Builder in ``generalised-grogu`` mode.  For the
    # fitting variant we initialize the two transverse directions explicitly;
    # this avoids relying on an internal mode-switching shortcut.
    if spin_model == "generalised-fit":
        reference_directions = [
            dict(o=np.array([1.0, 0.0, 0.0]), vw=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
            dict(o=np.array([0.0, 1.0, 0.0]), vw=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])),
            dict(o=np.array([0.0, 0.0, 1.0]), vw=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])),
        ]
    else:
        reference_directions = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
    simulation = grogupy.Builder(ref_xcf_orientations=reference_directions)
    simulation.add_hamiltonian(hamiltonian)
    simulation.add_kspace(grogupy.Kspace(kset=kmesh))
    simulation.add_contour(grogupy.Contour(emin=float(emin), eset=int(eset), esetp=int(esetp)))

    # These are documented Builder attributes; set them explicitly so the
    # output remains reproducible across grogupy minor releases.
    simulation.spin_model = spin_model
    simulation.apply_spin_model = True

    simulation.add_magnetic_entities(_entity_specs(groups, orbital_model))
    simulation.add_pairs(
        [dict(ai=i, aj=j, Ruc=[0, 0, 0]) for i, j, _ in PAIR_SPECS]
    )
    simulation.solve()

    magnopy_out = Path(magnopy_out)
    exchange_json = Path(exchange_json)
    # Save exactly to the requested path.  grogupy's top-level save_magnopy()
    # helper otherwise appends a version-specific suffix.
    magnopy_out.write_text(simulation.to_magnopy(comments=False))

    result = {
        "input": {
            "fdf": str(fdf_path.resolve()),
            "kmesh": kmesh,
            "emin_eV": float(emin),
            "eset": int(eset),
            "esetp": int(esetp),
            "scf_xcf_orientation": list(map(float, scf_axis)),
            "orbital_model": orbital_model,
            "spin_model": spin_model,
            "groups_siesta_1based": groups,
            "groups_grogupy_0based": {
                gid: [atom - 1 for atom in atoms] for gid, atoms in groups.items()
            },
            "assume_no_soc": bool(assume_no_soc),
            "convention": (
                "H = 1/2 sum_{i != j} e_i^T J_ij e_j + sum_i e_i^T K_i e_i; "
                "normalized spins; J values in meV."
            ),
        },
        "results": _extract_grogupy_results(simulation, assume_no_soc),
    }
    exchange_json.write_text(json.dumps(_as_jsonable(result), indent=2) + "\n")
    return result


def internal_self_test() -> None:
    """
    Unit/sanity tests independent of SIESTA or grogupy.

    They test:
      (i) parser and partition logic,
      (ii) sign/DM tensor decomposition,
      (iii) global-rotation invariance of an isotropic pair Hamiltonian.
    """
    synthetic_groups = {1: [1, 3], 2: [2, 4], 3: [5, 7], 4: [6, 8]}
    synthetic_spins = {
        1: AtomicSpin(1, 0, 4, 0.5, 0, 0, +0.5, "C"),
        2: AtomicSpin(2, 0, 4, 0.5, 0, 0, -0.5, "C"),
        3: AtomicSpin(3, 0, 4, 0.5, 0, 0, +0.5, "C"),
        4: AtomicSpin(4, 0, 4, 0.5, 0, 0, -0.5, "C"),
        5: AtomicSpin(5, 0, 4, 0.5, 0, 0, +0.5, "C"),
        6: AtomicSpin(6, 0, 4, 0.5, 0, 0, -0.5, "C"),
        7: AtomicSpin(7, 0, 4, 0.5, 0, 0, +0.5, "C"),
        8: AtomicSpin(8, 0, 4, 0.5, 0, 0, -0.5, "C"),
    }
    validation = validate_groups(synthetic_groups, synthetic_spins, range(1, 9))
    assert validation["partition_ok"]

    J0 = 7.3
    iso = J0 * np.eye(3)
    check = no_soc_tensor_check(iso, tolerance=1e-12)
    assert check["passes_isotropy_check"]
    assert abs(check["J_iso_meV"] - J0) < 1e-12

    # Known isotropic + DM tensor: D_z = 0.8 meV.
    test_tensor = np.array(
        [[J0, +0.8, 0.0], [-0.8, J0, 0.0], [0.0, 0.0, J0]], dtype=float
    )
    decomposition = decompose_exchange_tensor(test_tensor)
    assert np.allclose(decomposition["D_meV"], [0.0, 0.0, 0.8])
    assert not no_soc_tensor_check(test_tensor, tolerance=1e-8)["passes_isotropy_check"]

    # Relative angle, hence isotropic exchange energy, is invariant under a
    # common rotation of both spins.
    e1 = np.array([0.0, 0.0, 1.0])
    e2 = np.array([1.0, 0.0, 0.0])
    theta = np.pi / 3
    rotation_y = np.array(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ]
    )
    e_before = float(e1 @ iso @ e2)
    e_after = float((rotation_y @ e1) @ iso @ (rotation_y @ e2))
    assert abs(e_before - e_after) < 1e-12

    # All selected atoms must be converted exactly once to zero-based indices.
    specs = _entity_specs(synthetic_groups, "all")
    assert specs[0]["atom"] == [0, 2]
    print("Internal self-tests: PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate delocalized radical entities and run grogupy exchange extraction."
    )
    parser.add_argument("--out", help="SIESTA .out file for Mulliken-spin validation.")
    parser.add_argument(
        "--groups",
        required=False,
        help="Text file containing GROUP n lists in SIESTA 1-based atom numbering.",
    )
    parser.add_argument(
        "--report",
        default="radical_spin_report.json",
        help="JSON output path for the Mulliken entity report.",
    )
    parser.add_argument(
        "--run-grogupy",
        action="store_true",
        help="Run torque / Green-function exchange extraction through grogupy.",
    )
    parser.add_argument(
        "--fdf",
        help="Original SIESTA .fdf path (required with --run-grogupy).",
    )
    parser.add_argument(
        "--kmesh",
        nargs=3,
        type=int,
        default=[1, 1, 1],
        metavar=("NKX", "NKY", "NKZ"),
        help="Green-function k mesh. For a large isolated molecular supercell use 1 1 1.",
    )
    parser.add_argument(
        "--emin",
        type=float,
        default=None,
        help="Contour lower bound in eV. Default uses grogupy.automatic_emin().",
    )
    parser.add_argument(
        "--eset",
        type=int,
        default=400,
        help="Complex-contour sampling density. Converge it; start with 400.",
    )
    parser.add_argument(
        "--esetp",
        type=int,
        default=10000,
        help="Real-axis sampling parameter passed to grogupy.Contour.",
    )
    parser.add_argument(
        "--scf-axis",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 1.0],
        metavar=("X", "Y", "Z"),
        help=(
            "Reference axis for the nearly-collinear input exchange field. "
            "Use [0,0,1] for the supplied calculation."
        ),
    )
    parser.add_argument(
        "--orbital-model",
        choices=("all", "p"),
        default="all",
        help="Entity projector: all PAOs on each C cluster (baseline), or l=1 p shells.",
    )
    parser.add_argument(
        "--spin-model",
        choices=("generalised-fit", "generalised-grogu", "isotropic-only"),
        default="generalised-grogu",
        help=(
            "grogupy spin-model solver. generalised-grogu is the baseline; "
            "generalised-fit is useful only as a sensitivity check."
        ),
    )
    parser.add_argument(
        "--assume-no-soc",
        action="store_true",
        help="Apply isotropy / zero-DM diagnostics appropriate only when SOC is absent.",
    )
    parser.add_argument(
        "--magnopy",
        default="radical_exchange.magnopy",
        help="Output magnopy file from grogupy.",
    )
    parser.add_argument(
        "--exchange-json",
        default="radical_exchange.json",
        help="Machine-readable grogupy tensor summary.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run parser and physics unit tests without accessing SIESTA files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.self_test:
        internal_self_test()
        if not any((args.out, args.run_grogupy)):
            return 0

    if not args.groups and (args.out or args.run_grogupy):
        raise SystemExit("--groups is required when reading SIESTA data or running grogupy.")

    groups: dict[int, list[int]] | None = parse_groups(args.groups) if args.groups else None

    if args.out:
        atom_spins = parse_final_mulliken_table(args.out)
        report = spin_report(atom_spins, groups)  # type: ignore[arg-type]
        Path(args.report).write_text(json.dumps(_as_jsonable(report), indent=2) + "\n")
        print_spin_report(report)
        print(f"\nWrote spin report: {args.report}")

    if args.run_grogupy:
        if not args.fdf:
            raise SystemExit("--fdf is required with --run-grogupy.")
        result = run_grogupy(
            args.fdf,
            groups,  # type: ignore[arg-type]
            kmesh=args.kmesh,
            emin=args.emin,
            eset=args.eset,
            esetp=args.esetp,
            scf_axis=args.scf_axis,
            orbital_model=args.orbital_model,
            spin_model=args.spin_model,
            assume_no_soc=args.assume_no_soc,
            magnopy_out=args.magnopy,
            exchange_json=args.exchange_json,
        )
        print("\n=== grogupy exchange result ===")
        for pair in result["results"]["pairs"]:
            print(
                f"{pair['label']}: J_iso={pair['J_iso_meV']:+.6f} meV, "
                f"|D|={np.linalg.norm(pair['D_meV']):.3e} meV"
            )
            if "no_SOC_check" in pair:
                print(
                    "  no-SOC tensor check:",
                    "PASS" if pair["no_SOC_check"]["passes_isotropy_check"] else "FAIL",
                )
        print(f"Wrote magnopy model: {args.magnopy}")
        print(f"Wrote JSON tensor summary: {args.exchange_json}")

    if not (args.self_test or args.out or args.run_grogupy):
        build_parser().print_help()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
