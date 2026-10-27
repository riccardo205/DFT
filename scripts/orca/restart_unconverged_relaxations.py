#!/usr/bin/env python3
"""
Turn ORCA geometry optimisations that did not converge (e.g. killed by the walltime) into
restarts from the last point reached, by changing ONLY high_spin.inp; no file is renamed.

For every <root>/**/high_spin/high_spin.out without "THE OPTIMIZATION HAS CONVERGED":
  - "* xyzfile <charge> <mult> geometry.xyz" -> "... high_spin.xyz" (last geometry written
    by the previous run; ORCA reads it at startup and then keeps overwriting it);
  - only the geometry is carried over: submit_functional_scan.py archives (renames)
    high_spin.gbw on resubmission, so the SCF starts from ORCA's default guess at the
    last geometry; a MORead / %moinp left from an earlier restart attempt is removed.
Constraints and all other settings are unchanged. Already-prepared inputs are skipped.
Note: the new run overwrites high_spin_trj.xyz; copy it first if you want the full path.

Usage (standard library only, runs locally or on DelftBlue):
    python3 restart_unconverged_relaxations.py <run folder>          # dry run
    python3 restart_unconverged_relaxations.py <run folder> --yes    # rewrite the inputs
then push the .inp files and resubmit with submit_functional_scan.py (RESUBMIT = True).
"""

import argparse
import re
from pathlib import Path

CONVERGED = "THE OPTIMIZATION HAS CONVERGED"


def restart_input(text):
    text = re.sub(r"^(\*\s*xyzfile\s+-?\d+\s+\d+\s+)\S+", r"\1high_spin.xyz", text, flags=re.M | re.I)
    text = re.sub(r"^(!.*?)[ \t]+MORead\b", r"\1", text, flags=re.M | re.I)
    text = re.sub(r'^%moinp[^\n]*\n\n?', "", text, flags=re.M | re.I)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="run folder, e.g. .../dimers_run2/run4")
    parser.add_argument("--yes", action="store_true", help="rewrite the inputs (default: dry run)")
    args = parser.parse_args()

    changed, skipped = [], []
    for inp in sorted(args.root.rglob("high_spin/high_spin.inp")):
        folder = inp.parent
        label = folder.relative_to(args.root).as_posix()
        out = folder / "high_spin.out"
        if not out.exists():
            skipped.append(f"{label}: no high_spin.out (never ran)")
            continue
        out_text = out.read_text(errors="ignore")
        if CONVERGED in out_text:
            continue
        if not (folder / "high_spin.xyz").exists():
            skipped.append(f"{label}: missing high_spin.xyz")
            continue

        text = inp.read_text()
        new_text = restart_input(text)
        if new_text == text:
            skipped.append(f"{label}: already prepared")
            continue
        if args.yes:
            inp.write_text(new_text)
        cycles = out_text.count("GEOMETRY OPTIMIZATION CYCLE")
        changed.append(f"{label}: restart from high_spin.xyz (after cycle {cycles})")

    print(("REWRITTEN" if args.yes else "WOULD REWRITE") + f" ({len(changed)}):")
    for line in changed:
        print("  " + line)
    if skipped:
        print(f"SKIPPED ({len(skipped)}):")
        for line in skipped:
            print("  " + line)
    if not args.yes:
        print("\nDry run. Re-run with --yes to rewrite the inputs.")


if __name__ == "__main__":
    main()
