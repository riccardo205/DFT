#!/usr/bin/env python3

from pathlib import Path
import shutil
import sys


def find_out_files(root: Path):
    """
    Recursively find real ORCA job outputs under root: a .out file is only
    kept if a sibling .inp with the same stem exists. This filters out Slurm's
    own "slurm.out" logs and ORCA's auxiliary per-atom outputs (e.g.
    "high_spin_atom79.out").
    """
    return [
        out_file
        for out_file in sorted(root.rglob("*.out"))
        if out_file.with_suffix(".inp").exists()
    ]


def flat_name(out_file: Path, root: Path):
    """
    Build a unique, collision-free flat filename from the .out file's path
    relative to root, e.g.:
      field_z-0.005/broken_symmetry/broken_symmetry.out -> field_z-0.005_broken_symmetry.out
      L000_C020_R000/high_spin.out                      -> L000_C020_R000_high_spin.out
    """
    dirs = out_file.relative_to(root).parts[:-1]
    stem = out_file.stem
    if dirs and dirs[-1] == stem:
        return "_".join(dirs) + ".out"
    return "_".join((*dirs, stem)) + ".out"


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def main(root: Path, overwrite: bool):
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root.resolve()}")

    output_dir = root.parent / f"{root.name}_all_out_files"

    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists:\n  {output_dir.resolve()}\n\n"
            f"Set OVERWRITE = True only if you really want to regenerate it."
        )
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    out_files = find_out_files(root)
    if not out_files:
        raise RuntimeError(f"No ORCA .out/.inp pairs found under {root}")

    seen = {}
    for out_file in out_files:
        name = flat_name(out_file, root)
        if name in seen:
            raise RuntimeError(
                f"Name collision while flattening: {name!r} "
                f"(from {out_file} and {seen[name]})"
            )
        seen[name] = out_file

        shutil.copy2(out_file, output_dir / name)
        print(f"Copied: {out_file.relative_to(root)} -> {name}")

    print()
    print("=" * 70)
    print("Output files collected")
    print("=" * 70)
    print(f"Source : {root.resolve()}")
    print(f"Target : {output_dir.resolve()}")
    print(f"Files  : {len(out_files)}")


if __name__ == "__main__":
    # USER SETTINGS
    ROOT_DIR = REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scan"
    OVERWRITE = False

    if len(sys.argv) == 2:
        ROOT_DIR = Path(sys.argv[1])

    print(f"Collecting ORCA outputs from {ROOT_DIR}")
    main(ROOT_DIR, OVERWRITE)
