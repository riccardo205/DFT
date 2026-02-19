#!/usr/bin/env python3
from pathlib import Path
import shutil

FERRO_DIR_SUFFIX = "_ferro"
RADICAL_DIR_SUFFIX = "_4radical_2"
FERRO_FILE_SUFFIX = "_ferro.fdf"
RADICAL_FILE_SUFFIX = "_4radical_2.fdf"

def run(root: Path):
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return

    for src_dir in [p for p in root.iterdir() if p.is_dir() and p.name.endswith(FERRO_DIR_SUFFIX)]:
        dst_dir = src_dir.with_name(src_dir.name[:-len(FERRO_DIR_SUFFIX)] + RADICAL_DIR_SUFFIX)

        if dst_dir.exists():
            print(f"[skip] Destination exists: {dst_dir}")
            continue

        print(f"[copy] {src_dir} -> {dst_dir}")
        shutil.copytree(src_dir, dst_dir)

        # Rename any *_ferro.fdf files inside the cloned folder (recursively)
        for f in dst_dir.rglob(f"*{FERRO_FILE_SUFFIX}"):
            target = f.with_name(f.name[:-len(FERRO_FILE_SUFFIX)] + RADICAL_FILE_SUFFIX)
            if target.exists():
                print(f"[skip] Target exists: {target}")
                continue
            print(f"[rename] {f.relative_to(dst_dir)} -> {target.name}")
            f.rename(target)

# --- set your input folder here ---
ROOT = Path("/Users/riccardo/Documenti/TuDelft/phd/Analysis_Code/DFT/Short-2OS/runs/conformers/slow_roomT")
run(ROOT)