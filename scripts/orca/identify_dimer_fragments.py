#!/usr/bin/env python3
"""
Atom -> fragment assignment for every dimer of dimers_run2/<run>_broken_symmetry, with checks.

Each monomer (L135_C120_R090 bonding) is cut at its three linker bonds into four fragments,
named after the REFERENCE monomer numbering (angle_scan_relaxation/L135_C120_R090):

    T41 : terminal radical ring containing reference C41   (cut at bond 41-0)
    B9  : backbone half containing reference atom 9          (cut at bonds 41-0 and 9-17)
    B17 : backbone half containing reference atom 17         (cut at bonds 9-17 and 23-28)
    T28 : terminal radical ring containing reference C28   (cut at bond 23-28)

and grouped into the two radical halves H41 = T41 + B9 and H28 = T28 + B17 (one S = 1/2
each). The reference atoms are located in every monomer by graph isomorphism, so dimers
with a different atom numbering are handled.

Checks per dimer:
  1. atoms 0..N-1 and N..2N-1 are the two monomers (each connected, no bond between them)
  2. each monomer is isomorphic to the reference; the reference automorphisms are reported,
     to see whether "T41" vs "T28" is topologically distinguishable at all
  3. fragment sizes and element counts equal those of the reference fragments
  4. fragments are connected, disjoint and cover the monomer; exactly one bond joins
     T41-B9, B9-B17 and B17-T28 and there is no other inter-fragment bond
  5. radical_fragments.json: every label A/B/C/D equals one radical half; it is reported as
     monomer + its actual outer plane angle ("orth" > 67.5 deg, else "twist"), because the two
     halves are topologically equivalent (graph automorphism swaps C41 <-> C28), so only the
     geometry tells them apart
  6. every <config>.inp: the FlipSpin atoms lie only in the radicals that the folder name
     marks as down ('d'), never in an 'u' radical; the number of atoms per flipped radical
     and whether the list exceeds ORCA's 64-atom FlipSpin limit are reported

Writes <stage>/<dimer>/fragments4.json and <run>_broken_symmetry/fragment_check.csv.
Usage: python3 identify_dimer_fragments.py   (no ORCA, geometry only)
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
from ase.io import read
import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from prepare_dimers_broken_symmetry import RADICALS, bond_graph  # noqa: E402

RUN_ROOT = REPO_ROOT / "runs" / "orca" / "short-2OS" / "dimers_run2"
RUNS = ["run2_broken_symmetry", "run3_broken_symmetry"]
REF_XYZ = REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scans" / "angle_scan_relaxation" / "L135_C120_R090" / "high_spin.xyz"
LINKS = {"T41-B9": (41, 0), "B9-B17": (9, 17), "B17-T28": (23, 28)}   # reference numbering
SEEDS = {"T41": 41, "B9": 9, "B17": 17, "T28": 28}
HALVES = {"H41": ("T41", "B9"), "H28": ("T28", "B17")}
# outer torsion of each half (reference numbering): dihedral quartet and the two rings whose
# least-squares planes define the actual outer plane angle
OUTER = {"H41": {"quartet": (42, 41, 0, 5), "rings": ([41, 42, 45, 44, 43], [0, 1, 2, 3, 4, 5])},
         "H28": {"quartet": (21, 23, 28, 29), "rings": ([28, 29, 30, 31, 32], [23, 21, 25, 24, 26, 27])}}
LOCALIZED_ABOVE = 67.5    # outer plane angle (deg) above which a half is called "orthogonal/localized"
FLIP_LIMIT = 64
FLIP_RE = re.compile(r"^\s*FlipSpin\s+(.*)$", re.M)


def cut_fragments(graph, mapping):
    """Fragments (sorted local indices) after removing the three linker bonds."""
    cut = graph.copy()
    for a, b in LINKS.values():
        cut.remove_edge(mapping[a], mapping[b])
    return {name: sorted(nx.node_connected_component(cut, mapping[s])) for name, s in SEEDS.items()}


def plane_angle(pos, ring_a, ring_b):
    normals = [np.linalg.svd(pos[r] - pos[r].mean(0))[2][-1] for r in (ring_a, ring_b)]
    return float(np.degrees(np.arccos(np.clip(abs(normals[0] @ normals[1]), 0, 1))))


def main():
    ref = read(REF_XYZ)
    n = len(ref)
    rg = bond_graph(ref)
    ident = {i: i for i in range(n)}
    ref_frag = cut_fragments(rg, ident)
    ref_elem = {k: Counter(ref.get_chemical_symbols()[i] for i in v) for k, v in ref_frag.items()}
    autos = list(GraphMatcher(rg, rg, node_match=lambda a, b: a["z"] == b["z"]).isomorphisms_iter())
    swaps = sum(1 for m in autos if m[41] == 28)
    print(f"Reference {REF_XYZ.parent.name}: {n} atoms, fragments "
          + ", ".join(f"{k} {len(v)} ({dict(ref_elem[k])})" for k, v in ref_frag.items()))
    print(f"Reference graph automorphisms: {len(autos)}; of these, {swaps} exchange C41 <-> C28 "
          f"({'T41/T28 are topologically equivalent' if swaps else 'T41 and T28 are topologically distinct'})")
    print()

    all_ok = True
    for run in RUNS:
        rows = []
        for stage_dir in sorted(p for p in (RUN_ROOT / run).iterdir() if p.is_dir()):
            for dimer_dir in sorted((p for p in stage_dir.iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name)):
                xyz = dimer_dir / "uuuu" / "geometry.xyz"
                if not xyz.exists():
                    continue
                dimer = read(xyz)
                sym = dimer.get_chemical_symbols()
                g = bond_graph(dimer)
                problems = []

                # 1. monomers
                comps = [set(c) for c in nx.connected_components(g)]
                if sorted(map(sorted, comps)) != [list(range(n)), list(range(n, 2 * n))]:
                    problems.append(f"connected components are not the two index blocks ({[len(c) for c in comps]})")

                monomers, labels_to_half = {}, {}
                for k, off in enumerate((0, n)):
                    sub = nx.convert_node_labels_to_integers(g.subgraph(range(off, off + n)), ordering="sorted")
                    gm = GraphMatcher(rg, sub, node_match=lambda a, b: a["z"] == b["z"])
                    if not gm.is_isomorphic():
                        problems.append(f"monomer {k + 1} not isomorphic to reference")
                        continue
                    mp = gm.mapping
                    frag = {name: [i + off for i in v] for name, v in cut_fragments(sub, mp).items()}
                    # 3. sizes / elements
                    for name, atoms in frag.items():
                        if Counter(sym[i] for i in atoms) != ref_elem[name]:
                            problems.append(f"monomer {k + 1} {name}: composition differs from reference")
                    # 4. partition and inter-fragment bonds
                    allatoms = sorted(a for v in frag.values() for a in v)
                    if allatoms != list(range(off, off + n)):
                        problems.append(f"monomer {k + 1}: fragments overlap or miss atoms")
                    owner = {a: name for name, v in frag.items() for a in v}
                    inter = Counter(tuple(sorted((owner[a], owner[b]))) for a, b in g.subgraph(range(off, off + n)).edges
                                    if owner[a] != owner[b])
                    expected = Counter(tuple(sorted(p.split("-"))) for p in LINKS)
                    if inter != expected:
                        problems.append(f"monomer {k + 1}: inter-fragment bonds {dict(inter)}")
                    outer = {}
                    for h, spec in OUTER.items():
                        q = [mp[i] + off for i in spec["quartet"]]
                        rings = [[mp[i] + off for i in r] for r in spec["rings"]]
                        outer[h] = {"dihedral": round(float(dimer.get_dihedral(*q)), 2),
                                    "plane_angle": round(plane_angle(dimer.positions, *rings), 2),
                                    "radical_C": mp[41 if h == "H41" else 28] + off}
                    monomers[k + 1] = {
                        "fragments": frag,
                        "halves": {h: sorted(frag[a] + frag[b]) for h, (a, b) in HALVES.items()},
                        "outer": outer,
                        "key_atoms": {f"ref{r}": mp[r] + off for r in (0, 9, 17, 23, 28, 41)},
                    }

                # 5. radical_fragments.json
                rf_path = dimer_dir / "radical_fragments.json"
                rf = json.loads(rf_path.read_text())["fragments"] if rf_path.exists() else {}
                for lab, atoms in rf.items():
                    match = [(m, h) for m, d in monomers.items() for h, v in d["halves"].items() if sorted(atoms) == v]
                    if len(match) == 1:
                        m, h = match[0]
                        o = monomers[m]["outer"][h]
                        kind = "orth" if o["plane_angle"] > LOCALIZED_ABOVE else "twist"
                        labels_to_half[lab] = f"M{m}-{kind}({o['plane_angle']:.0f})"
                    else:
                        labels_to_half[lab] = "NO MATCH"
                    if len(match) != 1:
                        problems.append(f"json {lab} matches no radical half")

                # 6. FlipSpin lists
                flip_info = {}
                for cfg_dir in sorted(p for p in dimer_dir.iterdir() if p.is_dir() and re.fullmatch(r"[ud]{4}", p.name)):
                    inp = cfg_dir / f"{cfg_dir.name}.inp"
                    if not inp.exists():
                        continue
                    m = FLIP_RE.search(inp.read_text())
                    flip = {int(x) for x in m.group(1).replace(",", " ").split()} if m else set()
                    per = {lab: len(flip & set(rf[lab])) for lab in rf}
                    down = {lab for lab, s in zip(RADICALS, cfg_dir.name) if s == "d"}
                    stray = flip - set().union(*(set(rf[lab]) for lab in down)) if down else flip
                    if stray:
                        problems.append(f"{cfg_dir.name}: {len(stray)} flipped atoms outside the 'd' radicals")
                    missing = [lab for lab in down if per[lab] == 0]
                    if missing:
                        problems.append(f"{cfg_dir.name}: 'd' radical(s) {missing} have no flipped atom")
                    flip_info[cfg_dir.name] = {"n": len(flip), "per_radical": {k: v for k, v in per.items() if v},
                                               "over_limit": len(flip) > FLIP_LIMIT}

                out = {
                    "reference": str(REF_XYZ.relative_to(REPO_ROOT)),
                    "monomers": monomers,
                    "json_label_to_half": labels_to_half,
                    "flip_lists": flip_info,
                    "problems": problems,
                }
                (dimer_dir / "fragments4.json").write_text(json.dumps(out, indent=1))
                over = [c for c, v in flip_info.items() if v["over_limit"]]
                rows.append({
                    "stage": stage_dir.name, "dimer": dimer_dir.name,
                    "central_bond_M1": f"{monomers[1]['key_atoms']['ref9']}-{monomers[1]['key_atoms']['ref17']}" if 1 in monomers else "",
                    "A": labels_to_half.get("A", ""), "B": labels_to_half.get("B", ""),
                    "C": labels_to_half.get("C", ""), "D": labels_to_half.get("D", ""),
                    "flip_sizes": " ".join(f"{c}:{v['n']}" for c, v in flip_info.items() if v["n"]),
                    "over_64": " ".join(over),
                    "ok": not problems, "problems": "; ".join(problems),
                })
                all_ok &= not problems

        with (RUN_ROOT / run / "fragment_check.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"=== {run}: {len(rows)} dimers, {sum(r['ok'] for r in rows)} pass all checks")
        strip = lambda x: re.sub(r"\(\d+\)", "", x)
        label_patterns = Counter((strip(r["A"]), strip(r["B"]), strip(r["C"]), strip(r["D"])) for r in rows)
        for pat, cnt in label_patterns.items():
            print(f"    A,B,C,D = {', '.join(pat)} : {cnt} dimers")
        n_over = sum(bool(r["over_64"]) for r in rows)
        print(f"    dimers with a FlipSpin list > {FLIP_LIMIT} atoms: {n_over}; size patterns: "
              f"{dict(Counter(r['flip_sizes'] for r in rows).most_common(3))}")
        for r in rows:
            if not r["ok"]:
                print(f"    PROBLEM {r['stage']}/{r['dimer']}: {r['problems']}")
    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED (see above)")


if __name__ == "__main__":
    main()
