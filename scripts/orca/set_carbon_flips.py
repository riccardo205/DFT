#!/usr/bin/env python3
"""
Rewrite the FlipSpin lines of every dimers_run2/<run>_broken_symmetry input so that only the
CARBON atoms of the flipped radicals are listed (27 per radical: hydrogens carry < 3 % of the
radical |spin| and are left out, which keeps pair flips at 54 < 64 atoms, ORCA's limit).

For every dimer the atom assignment is rebuilt from the geometry, independently of the atom
numbering (the dimers do not all share one numbering):
  1. the two monomers are the two connected components of the bond graph;
  2. each monomer is mapped onto the reference L135_C120_R090 by graph isomorphism, which
     locates its radical carbons (reference C28, C41) and its three linker bonds;
  3. cutting the central bond gives the two radical halves; each half must contain exactly one
     radical carbon, 43 atoms, 27 of them carbons;
  4. labels follow the convention of prepare_dimers_broken_symmetry.py: A / B = halves of the
     monomer containing atom 0, around radical carbons 28 / 41; C / D = the other monomer's
     halves around 28+N / 41+N. The script checks that these four atoms are radical carbons;
  5. the halves must equal radical_fragments.json (the analysis scripts use that file).
Then, per configuration folder (u/d in the order A B C D): flip = carbons of the 'd' halves.
Checks: no hydrogen, no atom of an 'u' half, 27 atoms per flipped radical, FlipSpin < 64 atoms,
the folder's geometry.xyz equals the dimer geometry, exactly one FlipSpin line replaced, and
FinalMs equals (number of u - number of d) / 2.

Usage:  python3 set_carbon_flips.py            # dry run: checks and report only
        python3 set_carbon_flips.py --write    # also rewrite the FlipSpin lines
"""

import argparse
import filecmp
import json
import re
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
from ase.io import read
from networkx.algorithms.isomorphism import GraphMatcher

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from prepare_dimers_broken_symmetry import RADICALS, bond_graph  # noqa: E402

RUN_ROOT = REPO_ROOT / "runs" / "orca" / "short-2OS" / "dimers_run2"
RUNS = ["run2_broken_symmetry", "run3_broken_symmetry"]
REF_XYZ = REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans" / "angle_scan_relaxation" / "L135_C120_R090" / "high_spin.xyz"
REF_CENTRAL, REF_RADICALS = (9, 17), (28, 41)
FLIP_LIMIT = 64
FLIP_LINE = re.compile(r"^[ \t]*FlipSpin[^\n]*$", re.M)
FINALMS = re.compile(r"^\s*FinalMs\s+(-?\d+)", re.M)


def radical_halves(dimer, ref_graph):
    """{label: sorted atom list} for A-D, rebuilt from the bond graph, plus the radical carbons."""
    g = bond_graph(dimer)
    comps = sorted((sorted(c) for c in nx.connected_components(g)), key=min)
    if len(comps) != 2:
        raise ValueError(f"{len(comps)} molecules found, expected 2")
    halves, radical_c = {}, {}
    for k, comp in enumerate(comps):
        sub = g.subgraph(comp)
        gm = GraphMatcher(ref_graph, sub, node_match=lambda a, b: a["z"] == b["z"])
        if not gm.is_isomorphic():
            raise ValueError(f"monomer {k + 1} is not isomorphic to the reference")
        mp = gm.mapping                                   # reference index -> dimer index
        rad = {mp[r] for r in REF_RADICALS}               # the two radical carbons of this monomer
        cut = sub.copy()
        cut.remove_edge(mp[REF_CENTRAL[0]], mp[REF_CENTRAL[1]])
        local_anchor = (min(comp) + 28, min(comp) + 41)   # convention: atoms 28 / 41 (+N)
        if set(local_anchor) != rad:
            raise ValueError(f"monomer {k + 1}: atoms {local_anchor} are not its radical carbons {sorted(rad)}")
        for anchor, lab in zip(local_anchor, RADICALS[2 * k:2 * k + 2]):
            half = sorted(nx.node_connected_component(cut, anchor))
            if len(rad & set(half)) != 1:
                raise ValueError(f"half {lab} contains {len(rad & set(half))} radical carbons")
            halves[lab], radical_c[lab] = half, anchor
    return halves, radical_c


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="rewrite the FlipSpin lines")
    args = parser.parse_args()

    ref = read(REF_XYZ)
    ref_graph = bond_graph(ref)
    n_ref_c = Counter(ref.get_chemical_symbols())
    errors, totals, finished = [], Counter(), []
    for run in RUNS:
        dimers = sorted((d for d in (RUN_ROOT / run).glob("*/*") if (d / "uuuu" / "geometry.xyz").exists()),
                        key=lambda d: (d.parent.name, len(d.name), d.name))
        print(f"=== {run}: {len(dimers)} dimers")
        for d in dimers:
            label = f"{d.parent.name}/{d.name}"
            try:
                dimer = read(d / "uuuu" / "geometry.xyz")
                sym = dimer.get_chemical_symbols()
                if len(dimer) != 2 * len(ref) or Counter(sym) != Counter({k: 2 * v for k, v in n_ref_c.items()}):
                    raise ValueError("composition differs from 2 x reference")
                halves, radc = radical_halves(dimer, ref_graph)
                carbons = {lab: [a for a in h if sym[a] == "C"] for lab, h in halves.items()}
                for lab, h in halves.items():
                    if len(h) != 43 or len(carbons[lab]) != 27:
                        raise ValueError(f"{lab}: {len(h)} atoms / {len(carbons[lab])} C, expected 43 / 27")
                rf = json.loads((d / "radical_fragments.json").read_text())["fragments"]
                if any(sorted(rf[lab]) != halves[lab] for lab in RADICALS):
                    raise ValueError("halves differ from radical_fragments.json")
            except Exception as exc:                       # report and go on with the next dimer
                errors.append(f"{run}/{label}: {exc}")
                print(f"  {label}: ERROR {exc}")
                continue

            line = []
            for cfg in sorted(p for p in d.iterdir() if p.is_dir() and re.fullmatch(r"[ud]{4}", p.name)):
                inp = cfg / f"{cfg.name}.inp"
                if not inp.exists():
                    continue
                if not filecmp.cmp(cfg / "geometry.xyz", d / "uuuu" / "geometry.xyz", shallow=False):
                    errors.append(f"{run}/{label}/{cfg.name}: geometry.xyz differs from uuuu")
                text = inp.read_text()
                down = [lab for lab, s in zip(RADICALS, cfg.name) if s == "d"]
                if not down:                                # uuuu: must not flip anything
                    if FLIP_LINE.search(text):
                        errors.append(f"{run}/{label}/uuuu: has a FlipSpin line")
                    continue
                flip = sorted(a for lab in down for a in carbons[lab])
                up_atoms = {a for lab in RADICALS if lab not in down for a in halves[lab]}
                ms = FINALMS.search(text)
                expected_ms = (cfg.name.count("u") - cfg.name.count("d")) // 2
                checks = {
                    "hydrogen flipped": any(sym[a] != "C" for a in flip),
                    "atom of an 'u' radical flipped": bool(set(flip) & up_atoms),
                    "not 27 C per flipped radical": len(flip) != 27 * len(down),
                    f"FlipSpin >= {FLIP_LIMIT}": len(flip) >= FLIP_LIMIT,
                    "FinalMs wrong": not ms or int(ms.group(1)) != expected_ms,
                }
                bad = [k for k, v in checks.items() if v]
                if bad:
                    errors.append(f"{run}/{label}/{cfg.name}: {', '.join(bad)}")
                new_text, n_sub = FLIP_LINE.subn(f"    FlipSpin {', '.join(map(str, flip))}", text)
                if n_sub != 1:
                    errors.append(f"{run}/{label}/{cfg.name}: {n_sub} FlipSpin lines")
                    continue
                out = cfg / f"{cfg.name}.out"
                if out.exists() and "ORCA TERMINATED NORMALLY" in out.read_text(errors="ignore"):
                    finished.append(f"{run}/{label}/{cfg.name}")
                if args.write and new_text != text and not bad:
                    inp.write_text(new_text)
                    totals["rewritten"] += 1
                totals[len(flip)] += 1
                line.append(f"{cfg.name}:{len(flip)}")
            print(f"  {label:18s} radical C: A={radc['A']} B={radc['B']} C={radc['C']} D={radc['D']}  "
                  f"flips {' '.join(line)}")

    print()
    print(f"FlipSpin list sizes: {dict(sorted((k, v) for k, v in totals.items() if isinstance(k, int)))}")
    if args.write:
        print(f"inputs rewritten: {totals['rewritten']}")
    if finished:
        print(f"note: {len(finished)} configurations already have a finished output from the old flip list: "
              + ", ".join(finished))
    print("ALL CHECKS PASSED" if not errors else f"{len(errors)} PROBLEMS:\n  " + "\n  ".join(errors))


if __name__ == "__main__":
    main()
