#!/usr/bin/env python3
"""
PBE0 broken-symmetry single points (FlipSpin / FinalMs) on the four-radical dimers of
dimers_run2/run2 and dimers_run2/run3, for fitting the pairwise J_ij of a four-spin
Heisenberg model.

Radical centres. Each monomer is a diradical whose two S=1/2 spins sit on the two halves
obtained by cutting its central C-C bond: in the PBE0 high-spin L135_C120_R090 monomer
each half carries 1.00 spin, centred on atom 28 (0.62) and atom 41 (0.55). The run2 dimers
mix two atom numberings of the same molecule (central bond 9-17 as in the reference
monomer, or 3-14), but in both the radical carbons are atoms 28 and 41 (the SIESTA
DM.InitSpin atoms 29, 42, 115, 128). So, for every dimer separately, each monomer is mapped
onto the reference monomer by graph isomorphism to find its central bond, and
    A = half of monomer 1 containing atom 28      B = half of monomer 1 containing atom 41
    C = half of monomer 2 containing atom 28+N    D = half of monomer 2 containing atom 41+N
FlipSpin flips the carbon atoms of each half (RADICAL_ATOMS = "carbons", 27 per radical; the
hydrogens carry < 3 % of the radical |spin|), or the whole half ("fragment"); with a float
threshold only the atoms whose reference HS spin population exceeds it are flipped. The atom lists are
written to <stage>/<dimer>/radical_fragments.json and reused by
collect_dimers_broken_symmetry.py.

All configurations start from the quintet (* xyzfile 0 5); FinalMs sets the target Ms.
ORCA atom numbering in FlipSpin is 0-based.

Geometry stages:
    initial/  : starting geometry of the relaxation (<dimer>/high_spin/geometry.xyz)
    relaxed/  : latest relaxation geometry (high_spin.xyz), converged or not; unconverged
                relaxations are listed in relaxation_status.csv and printed at the end
Writes <run>_broken_symmetry/<stage>/<dimer>/<config>/{<config>.inp, geometry.xyz}; folder
names use u/d (duuu = "-+++"). Re-running leaves existing folders alone, except that a
folder whose geometry changed (relaxation continued) and that has no output yet is
refreshed with the new geometry.
Natural orbitals: every job writes UHF natural orbitals and corresponding orbitals (! UNO UCO)
and prints the UNO occupation numbers.

Restart from an earlier high-spin run (MOREAD_FROM): every configuration, uuuu included, reads
<run>_broken_symmetry_high_spin/<stage>/<dimer>/uuuu/uuuu.gbw (! MORead NoAutoStart, relative
%moinp path; orca_job.sh runs ORCA in the job folder). That tree holds the earlier uuuu runs, so
the jobs stay independent: the new uuuu only adds the UNO/UCO analysis on a converged wavefunction,
and the FlipSpin high-spin stage starts converged. Runs set to None start from scratch.
REWRITE_INPUTS = True rewrites the .inp of existing folders in place (geometry.xyz and any
outputs are left untouched, so the inputs stay consistent with the geometry of that .gbw);
existing FlipSpin / FinalMs lines are copied verbatim, so hand-edited flip lists survive.

All jobs are independent: submit with submit_functional_scan.py.
"""

import csv
import filecmp
import json
import re
import shutil
from pathlib import Path

import networkx as nx
import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs
from networkx.algorithms.isomorphism import GraphMatcher

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

RADICALS = "ABCD"
# configuration -> FinalMs; "+" = up, "-" = down, in the order A B C D
CONFIGURATIONS = {
    "++++": 2,
    "-+++": 1,
    "+-++": 1,
    "++-+": 1,
    "+++-": 1,
    "--++": 0,
    "-+-+": 0,
    "-++-": 0,
}

TEMPLATE = """! {functional} {basis} TightSCF UKS DEFGRID3{moread} UNO UCO
{moinp}
%scf
    MaxIter {max_iter}
{conv_check}{flip}end

%output
    Print[P_UNO_OccNum] 1
end

%pal
    nprocs {nprocs}
end

%maxcore {maxcore}

* xyzfile {charge} {mult} geometry.xyz
"""


def folder_name(config):
    return config.replace("+", "u").replace("-", "d")


def bond_graph(atoms):
    nl = NeighborList(natural_cutoffs(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)
    graph = nx.Graph()
    graph.add_nodes_from((i, {"z": int(z)}) for i, z in enumerate(atoms.numbers))
    graph.add_edges_from((int(i), int(j)) for i, j in zip(*nl.get_connectivity_matrix().nonzero()) if i < j)
    return graph


def monomer_radical_halves(monomer, ref_graph, ref_central_bond, anchors, ref_selection):
    """Atom lists (local indices) of the halves containing each anchor, and the central bond used."""
    graph = bond_graph(monomer)
    matcher = GraphMatcher(ref_graph, graph, node_match=lambda a, b: a["z"] == b["z"])
    if not matcher.is_isomorphic():
        raise ValueError("monomer bonding differs from the reference monomer")
    mapping = matcher.mapping                              # reference atom -> this monomer's atom
    bond = tuple(mapping[i] for i in ref_central_bond)
    cut = graph.copy()
    cut.remove_edge(*bond)
    halves = [sorted(nx.node_connected_component(cut, a)) for a in anchors]
    if set(halves[0]) & set(halves[1]) or len(halves[0]) + len(halves[1]) != len(monomer):
        raise ValueError(f"anchors {anchors} are not on the two halves of central bond {bond}")
    if ref_selection is not None:                          # keep atoms with |HS spin| > threshold
        keep = {mapping[i] for i in ref_selection}
        halves = [[i for i in half if i in keep] for half in halves]
    return halves, bond


def mulliken_spins(out_file):
    text = Path(out_file).read_text(encoding="utf-8", errors="ignore")
    block = text.split("MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS")[-1].split("Sum of atomic")[0]
    return {int(m.group(1)): float(m.group(2))
            for m in re.finditer(r"^\s*(\d+)\s+\w+\s*:\s*-?\d+\.\d+\s+(-?\d+\.\d+)", block, re.M)}


def dimer_fragments(dimer, n_monomer, ref_graph, ref_central_bond, anchors, ref_selection):
    fragments, bonds = {}, []
    for k, offset in enumerate((0, n_monomer)):
        halves, bond = monomer_radical_halves(dimer[offset:offset + n_monomer], ref_graph,
                                              ref_central_bond, anchors, ref_selection)
        bonds.append([b + offset for b in bond])
        for h, half in enumerate(halves):
            fragments[RADICALS[2 * k + h]] = [i + offset for i in half]
    return fragments, bonds


def relaxation_state(out_file):
    if not out_file.exists():
        return "no_output"
    text = out_file.read_text(encoding="utf-8", errors="ignore")
    if "ORCA TERMINATED NORMALLY" in text and "THE OPTIMIZATION HAS CONVERGED" in text:
        return "converged"
    return "not_converged"


def main():
    # ---- USER SETTINGS ----
    RUN_ROOT = REPO_ROOT / "runs" / "orca" / "short-2OS" / "dimers_run2"
    JOBS = {                                     # source run -> geometry stages
        "run2": ["relaxed"],
        "run3": ["initial", "relaxed"],
    }

    REF_MONOMER_XYZ = (REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans" / "angle_scan_relaxation"
                       / "L135_C120_R090" / "high_spin.xyz")
    REF_HS_SPIN_OUT = (REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans" / "angle_scan_exchange_analysis"
                       / "L135_C120_R090" / "high_spin" / "high_spin.out")   # only used for a float RADICAL_ATOMS
    REF_CENTRAL_BOND = (9, 17)
    RADICAL_ANCHORS = (28, 41)       # radical carbons of every monomer (A/C and B/D)
    RADICAL_ATOMS = "carbons"        # "carbons" = carbon atoms of the half (27; pair flips 54 < 64, the
                                     # ORCA FlipSpin limit); "fragment" = whole half (43; pair flips 86,
                                     # truncated by ORCA); float = |HS spin| threshold

    FUNCTIONAL = "PBE0"
    BASIS = "def2-TZVP"
    CHARGE = 0
    MULTIPLICITY = 5                 # quintet reference for every configuration
    NPROCS = 16
    MAXCORE_MB = 2000
    MAX_ITER = 500
    CONV_CHECK_MODE_0 = False        # True stalled the FlipSpin high-spin stage for ~55 SCF cycles (noise
                                     # ~1e-7 Eh never meets all criteria at once); False = ORCA default
    MOREAD_FROM = {                  # run -> tree with converged uuuu.gbw files, or None (start from scratch)
        "run2": "run2_broken_symmetry_high_spin",
        "run3": "run3_broken_symmetry_high_spin",
    }
    REWRITE_INPUTS = True            # rewrite .inp of existing folders (geometry/outputs untouched)

    # ---- SETUP ----
    ref = read(REF_MONOMER_XYZ)
    n_monomer = len(ref)
    ref_graph = bond_graph(ref)
    ref_selection = None
    if RADICAL_ATOMS not in ("fragment", "carbons"):
        spins = mulliken_spins(REF_HS_SPIN_OUT)
        ref_selection = [i for i, s in spins.items() if abs(s) > RADICAL_ATOMS]

    not_converged = []
    for run, stages in JOBS.items():
        source_root = RUN_ROOT / run
        output_root = RUN_ROOT / f"{run}_broken_symmetry"
        output_root.mkdir(exist_ok=True)
        dimer_dirs = sorted((d for d in source_root.iterdir() if (d / "high_spin").is_dir()),
                            key=lambda d: (len(d.name), d.name))

        status_rows = []
        for dimer_dir in dimer_dirs:
            state = relaxation_state(dimer_dir / "high_spin" / "high_spin.out")
            status_rows.append({"dimer": dimer_dir.name, "relaxation": state})
            if state != "converged":
                not_converged.append(f"{run}/{dimer_dir.name} ({state})")
        with (output_root / "relaxation_status.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=status_rows[0].keys())
            writer.writeheader()
            writer.writerows(status_rows)
        relax_state = {r["dimer"]: r["relaxation"] for r in status_rows}

        manifest_rows, created, refreshed, stale, rewritten = [], 0, 0, [], 0
        for stage in stages:
            for dimer_dir in dimer_dirs:
                relax = dimer_dir / "high_spin"
                source_xyz = relax / ("geometry.xyz" if stage == "initial" else "high_spin.xyz")
                if not source_xyz.exists():
                    print(f"SKIP (no {source_xyz.name}): {run}/{dimer_dir.name}")
                    continue
                dimer = read(source_xyz)
                if list(dimer.numbers) != list(ref.numbers) * 2:
                    raise ValueError(f"{source_xyz}: element order differs from 2 x reference monomer")
                fragments, central_bonds = dimer_fragments(dimer, n_monomer, ref_graph, REF_CENTRAL_BOND,
                                                           RADICAL_ANCHORS, ref_selection)
                dimer_out = output_root / stage / dimer_dir.name
                dimer_out.mkdir(parents=True, exist_ok=True)
                (dimer_out / "radical_fragments.json").write_text(json.dumps({
                    "fragments": fragments,
                    "configurations": CONFIGURATIONS,
                    "central_bonds": central_bonds,
                    "anchors": RADICAL_ANCHORS,
                    "radical_atoms": RADICAL_ATOMS,
                    "source_geometry": str(source_xyz.relative_to(REPO_ROOT)),
                    "relaxation": relax_state[dimer_dir.name],
                }, indent=1))

                for config, final_ms in CONFIGURATIONS.items():
                    down = [r for r, s in zip(RADICALS, config) if s == "-"]
                    flip_atoms = sorted(a for r in down for a in fragments[r]
                                        if RADICAL_ATOMS != "carbons" or dimer.numbers[a] == 6)
                    name = folder_name(config)
                    run_folder = dimer_out / name
                    manifest_rows.append({
                        "stage": stage,
                        "dimer": dimer_dir.name,
                        "relaxation": relax_state[dimer_dir.name],
                        "configuration": config,
                        "folder": name,
                        "flipped_radicals": "".join(down),
                        "FinalMs": final_ms,
                        "n_flipped_atoms": len(flip_atoms),
                        "source_geometry": str(source_xyz.relative_to(REPO_ROOT)),
                        "input_file": f"{stage}/{dimer_dir.name}/{name}/{name}.inp",
                    })
                    copy_geometry = True
                    if run_folder.exists():
                        same = filecmp.cmp(source_xyz, run_folder / "geometry.xyz", shallow=False)
                        if REWRITE_INPUTS:
                            copy_geometry = False
                            rewritten += 1
                            if not same:
                                stale.append(f"{stage}/{dimer_dir.name}/{name} (input rewritten, old geometry kept)")
                        elif same:
                            continue
                        elif (run_folder / f"{name}.out").exists():
                            stale.append(f"{stage}/{dimer_dir.name}/{name}")
                            continue
                        else:
                            refreshed += 1
                    else:
                        run_folder.mkdir()
                        created += 1
                    if copy_geometry:
                        shutil.copy2(source_xyz, run_folder / "geometry.xyz")
                    moread, moinp = "", ""
                    if MOREAD_FROM.get(run):
                        gbw = f"../../../../{MOREAD_FROM[run]}/{stage}/{dimer_dir.name}/uuuu/uuuu.gbw"
                        moread, moinp = " MORead NoAutoStart", f'\n%moinp "{gbw}"\n'
                    manifest_rows[-1]["moread_from"] = MOREAD_FROM.get(run) and gbw or ""
                    flip = ""
                    old_inp = run_folder / f"{name}.inp"
                    old_flip = re.findall(r"^\s*(?:FlipSpin|FinalMs)\b.*$", old_inp.read_text(), re.M) \
                        if (REWRITE_INPUTS and old_inp.exists()) else []
                    if old_flip:                   # keep hand-edited FlipSpin / FinalMs lines as they are
                        flip = "\n" + "".join(f"    {line.strip()}\n" for line in old_flip)
                    elif flip_atoms:
                        flip = (f"\n    FlipSpin {', '.join(map(str, flip_atoms))}\n"
                                f"    FinalMs {final_ms}\n")
                    (run_folder / f"{name}.inp").write_text(TEMPLATE.format(
                        functional=FUNCTIONAL, basis=BASIS, max_iter=MAX_ITER, moread=moread, moinp=moinp,
                        conv_check="    ConvCheckMode 0\n" if CONV_CHECK_MODE_0 else "", flip=flip,
                        nprocs=NPROCS, maxcore=MAXCORE_MB, charge=CHARGE, mult=MULTIPLICITY,
                    ), encoding="utf-8")

        with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"{output_root.name}: {len(manifest_rows)} jobs in manifest, {created} new, "
              f"{refreshed} refreshed with a newer geometry, {rewritten} inputs rewritten in place")
        for path in stale:
            print(f"  WARNING geometry changed but output exists, left alone: {path}")

    print()
    print("=" * 70)
    print("Dimer broken-symmetry single points (FlipSpin) prepared")
    print("=" * 70)
    print(f"Relaxations NOT converged ({len(not_converged)}), prepared on their last geometry:")
    for item in not_converged:
        print(f"  {item}")


if __name__ == "__main__":
    main()
