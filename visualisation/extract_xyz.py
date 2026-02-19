#!/usr/bin/env python3
"""
Collect every *.xyz file that sits one level below <source_root>,
rename each file to <subfolder>.xyz, and copy it into <target_root>.

Usage:
    python collect_xyz.py <source_root> <target_root>
"""

import sys
from pathlib import Path
import shutil

def main(source_root: Path, target_root: Path) -> None:
    if not source_root.is_dir():
        sys.exit(f"Source folder {source_root} doesn’t exist or isn’t a directory.")

    target_root.mkdir(parents=True, exist_ok=True)

    for subdir in source_root.iterdir():
        if not subdir.is_dir():
            continue  # skip files at the top level

        # look for the first *.xyz in that subfolder (adjust if you expect many)
        xyz_files = list(subdir.glob("*.xyz"))
        if not xyz_files:
            print(f"[skip] no .xyz found in {subdir}")
            continue

        src = xyz_files[0]          # pick the first one (or loop if you expect more)
        dst = target_root / f"{subdir.name}.xyz"

        # If a file with that name already exists, decide what you want to do.
        # Here we overwrite; comment out to skip instead.
        shutil.copy2(src, dst)
        print(f"[ok] {src.relative_to(source_root)} ➜ {dst.relative_to(target_root)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python collect_xyz.py <source_root> <target_root>")

    source = Path(sys.argv[1]).expanduser().resolve()
    target = Path(sys.argv[2]).expanduser().resolve()
    main(source, target)