#!/usr/bin/env python3
"""
Choose reduced FlipSpin atom lists for the dimer broken-symmetry single points.

ORCA keeps only the first 64 atoms of a FlipSpin list, so flipping two whole 43-atom
radical halves (86 atoms) silently breaks the guess (and crashed conf_1 dduu/dudu/duud).
Instead of a fixed spin-population threshold, for every radical fragment the atoms are
taken in order of decreasing |Mulliken spin| of the converged high-spin (uuuu) state of
the same dimer, until TARGET of the fragment's total |spin| is captured. Each fragment is
capped at MAX_FLIP_ATOMS / 2 atoms, so a pair flip never exceeds MAX_FLIP_ATOMS (< 64).

Usage (from the repo root or anywhere):
    python3 select_flip_atoms.py [DIMER_DIR ...]            # report only
    python3 select_flip_atoms.py [DIMER_DIR ...] --write    # also rewrite FlipSpin lines
    python3 select_flip_atoms.py DIMER_DIR --write --configs dduu   # one configuration only
    python3 select_flip_atoms.py ... --write --moread   # also start from uuuu.gbw
With no DIMER_DIR, every <run>_broken_symmetry/<stage>/<dimer> under dimers_run2 is used.

High-spin spins: read from <dimer>/uuuu/uuuu.out, or, if that has not terminated normally,
from the same dimer in <run>_broken_symmetry_high_spin/ (where the converged uuuu runs are
kept). Dimers with no terminated uuuu in either place are skipped.

Radicals: A, B, C, D are the four radical halves of radical_fragments.json, centred on the
radical carbons 28, 41, 28+N, 41+N (checked for every dimer by identify_dimer_fragments.py).
--threshold S selects instead every atom of the radical with |spin| > S (still capped).

--write replaces only the "FlipSpin ..." line of the flip inputs whose run has not
terminated normally; every other setting is left as it is. With --moread the inputs also
get "! ... MORead NoAutoStart" and %moinp "uuuu.gbw" (the converged high-spin wavefunction, copied into
each configuration folder by bash_scripts/copy_uuuu_gbw.sh). The selection is saved to
<dimer>/flip_atoms.json.
"""

import argparse
import json
import re
from pathlib import Path

from prepare_dimers_broken_symmetry import RADICALS, mulliken_spins

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_ROOT = REPO_ROOT / "runs" / "orca" / "short-2OS" / "dimers_run2"


def terminated(out_file):
    return out_file.exists() and "ORCA TERMINATED NORMALLY" in out_file.read_text(errors="ignore")


def add_moread(text, gbw="uuuu.gbw"):
    """Add MORead and NoAutoStart to the keyword line and a %moinp block before %scf
    (if not there yet). NoAutoStart keeps ORCA from restarting from a leftover <config>.gbw."""
    for keyword in ("MORead", "NoAutoStart"):
        if not re.search(rf"^!.*\b{keyword}\b", text, flags=re.M | re.I):
            text = re.sub(r"^(!.*?)[ \t]*$", rf"\1 {keyword}", text, count=1, flags=re.M)
    if "%moinp" not in text.lower():
        text = re.sub(r"^%scf", f'%moinp "{gbw}"\n\n%scf', text, count=1, flags=re.M | re.I)
    return text


def select(atoms, spins, target, cap, threshold=None):
    """Fewest atoms (largest |spin| first) capturing `target` of sum|spin|, at most `cap`;
    with `threshold`, every atom with |spin| > threshold (largest first, at most `cap`)."""
    ranked = sorted(atoms, key=lambda a: -abs(spins[a]))
    total = sum(abs(spins[a]) for a in atoms)
    chosen, captured = [], 0.0
    for a in ranked:
        if len(chosen) == cap:
            break
        if threshold is None and captured >= target * total:
            break
        if threshold is not None and abs(spins[a]) <= threshold:
            break
        chosen.append(a)
        captured += abs(spins[a])
    return sorted(chosen), captured / total, total


def high_spin_out(dimer_dir):
    """uuuu.out of this dimer, or of the same dimer in <run>_broken_symmetry_high_spin."""
    local = dimer_dir / "uuuu" / "uuuu.out"
    if terminated(local):
        return local
    run_dir = dimer_dir.parent.parent
    alt = run_dir.parent / f"{run_dir.name}_high_spin" / dimer_dir.parent.name / dimer_dir.name / "uuuu" / "uuuu.out"
    return alt if terminated(alt) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dimer_dirs", nargs="*", type=Path)
    parser.add_argument("--target", type=float, default=0.95, help="fraction of fragment |spin| (default 0.95)")
    parser.add_argument("--max-flip-atoms", type=int, default=60, help="max atoms in one FlipSpin list (default 60)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="select atoms with |spin| > THRESHOLD instead of the --target fraction")
    parser.add_argument("--configs", nargs="*", help="only these configuration folders, e.g. dduu dudu")
    parser.add_argument("--write", action="store_true", help="rewrite the FlipSpin lines")
    parser.add_argument("--moread", action="store_true", help='also add MORead NoAutoStart / %%moinp "uuuu.gbw"')
    args = parser.parse_args()

    cap = args.max_flip_atoms // 2               # at most two radicals are flipped together
    dimer_dirs = args.dimer_dirs or sorted(
        (d for d in DEFAULT_ROOT.glob("*_broken_symmetry/*/*") if (d / "radical_fragments.json").exists()),
        key=lambda d: (str(d.parent), len(d.name), d.name)
    )

    for dimer_dir in dimer_dirs:
        dimer_dir = dimer_dir.resolve()
        label = str(dimer_dir.relative_to(DEFAULT_ROOT)) if DEFAULT_ROOT in dimer_dir.parents else str(dimer_dir)
        hs_out = high_spin_out(dimer_dir)
        if hs_out is None:
            print(f"== {label}: SKIP, no terminated uuuu (here or in the _high_spin tree)\n")
            continue

        setup = json.loads((dimer_dir / "radical_fragments.json").read_text())
        spins = mulliken_spins(hs_out)
        selected, report = {}, {}
        rule = f"|spin| > {args.threshold}" if args.threshold is not None else f"{args.target:.0%} of |spin|"
        print(f"== {label}  ({rule}, max {cap} atoms per radical; spins from {hs_out.relative_to(DEFAULT_ROOT)})")
        n_mon = len(spins) // 2
        for k, r in enumerate(RADICALS):
            atoms = setup["fragments"][r]
            centre = (28, 41)[k % 2] + n_mon * (k // 2)
            if centre not in atoms:
                raise RuntimeError(f"{label}: radical {r} does not contain its centre atom {centre}")
            chosen, frac, total = select(atoms, spins, args.target, cap, args.threshold)
            selected[r] = chosen
            net = sum(spins[a] for a in atoms)
            net_sel = sum(spins[a] for a in chosen)
            report[r] = {"centre": centre, "n_atoms": len(chosen), "captured_abs_spin": round(frac, 4),
                         "captured_net_spin": round(net_sel, 4),
                         "fragment_net_spin": round(net, 4), "fragment_abs_spin": round(total, 4)}
            flag = "" if (args.threshold is not None or frac >= args.target) else f"   WARNING: capped at {cap}, below target"
            print(f"  {r} (centre {centre:3d}): {len(chosen):2d} atoms, captured |spin| = {frac:.2f}, "
                  f"net spin {net_sel:+.2f} of {net:+.2f}{flag}")

        changes = []
        for config_dir in sorted(p for p in dimer_dir.iterdir() if p.is_dir() and p.name != "uuuu"):
            name = config_dir.name
            if args.configs and name not in args.configs:
                continue
            inp = config_dir / f"{name}.inp"
            if not inp.exists():
                continue
            down = [r for r, c in zip(RADICALS, name) if c == "d"]
            flip = sorted(a for r in down for a in selected[r])
            if len(flip) >= 64:
                raise RuntimeError(f"{inp}: {len(flip)} FlipSpin atoms, ORCA keeps only 64")
            text = inp.read_text()
            new_line = f"    FlipSpin {', '.join(map(str, flip))}"
            new_text, n_sub = re.subn(r"^[ \t]*FlipSpin[^\n]*$", new_line, text, flags=re.M)
            if n_sub != 1:
                raise RuntimeError(f"{inp}: expected one FlipSpin line, found {n_sub}")
            if args.moread:
                new_text = add_moread(new_text)
            if terminated(config_dir / f"{name}.out"):
                status = "finished, not changed"
            elif new_text == text:
                status = "already up to date"
            elif args.write:
                inp.write_text(new_text)
                status = "REWRITTEN"
            else:
                status = "would rewrite"
            changes.append(f"  {name} -> flip {' + '.join(down)} -> {len(flip)} atoms   [{status}]")
        print("\n".join(changes))

        if args.write:
            (dimer_dir / "flip_atoms.json").write_text(json.dumps({
                "source": f"Mulliken spin of {hs_out.relative_to(DEFAULT_ROOT)}",
                "target_abs_spin_fraction": None if args.threshold is not None else args.target,
                "abs_spin_threshold": args.threshold,
                "max_atoms_per_fragment": cap,
                "selected": selected,
                "report": report,
            }, indent=1))
        print()


if __name__ == "__main__":
    main()
