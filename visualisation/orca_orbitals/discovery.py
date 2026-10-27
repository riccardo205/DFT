"""
Find ORCA calculations below a root directory without assuming a folder layout.

A calculation is an ORCA text output (recognised by its banner, so slurm.out and
similar are ignored). Its companion files are matched strictly by base name in
the SAME directory, where the base name is the ORCA job name read from the
"NAME = <job>.inp" line of the output (ORCA names .gbw/.uno/.uco after the input,
not after the redirected .out file). Nothing is matched by pattern or proximity:
a guess.gbw, high_spin_reference.gbw or a .gbw in another folder is never used.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

ORCA_BANNER = "O   R   C   A"
NAME_RE = re.compile(r"^NAME\s*=\s*(\S+)", re.M)


@dataclass
class Calculation:
    out_path: Path
    directory: Path
    base: str                          # ORCA job name (input stem)
    gbw: Path
    uno: Path | None = None
    uco: Path | None = None
    property_txt: Path | None = None
    existing_cubes: list = field(default_factory=list)
    output_dir: Path | None = None

    @property
    def name(self):
        return self.base

    def label(self, root):
        try:
            rel = self.directory.relative_to(root)
        except ValueError:
            rel = self.directory
        rel = str(rel) if str(rel) != "." else self.directory.name
        return f"{rel}/{self.base}"


def is_orca_output(path, nbytes=65536):
    try:
        with open(path, errors="replace") as fh:
            return ORCA_BANNER in fh.read(nbytes)
    except OSError:
        return False


def _job_name(out_path):
    """Input stem from 'NAME = x.inp' (searched in the first part of the file), else None."""
    with open(out_path, errors="replace") as fh:
        head = fh.read(400_000)
    m = NAME_RE.search(head)
    return Path(m[1]).stem if m else None


def discover(root, recursive=True, output_dir_name="orbital_analysis", ignore_dir_names=(), include=None,
             central_output=False):
    """
    Returns (calculations, problems) where problems is a list of (path, reason).

    `include`: optional callable(Path) -> bool applied to each .out path.
    `central_output`: put each job's results in <root>/<output_dir_name>/<relative dir>/
    instead of <job dir>/<output_dir_name>/, keeping the calculation folders untouched.
    """
    root = Path(root).resolve()
    ignore = set(ignore_dir_names) | {output_dir_name, "__pycache__"}
    outs = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in ignore and not d.startswith(".") and not d.startswith("_orca_plot_work"))
        if not recursive:
            dirnames[:] = []
        for f in sorted(filenames):
            if f.endswith(".out"):
                p = Path(dirpath) / f
                if include is None or include(p):
                    outs.append(p)

    calcs, problems = [], []
    for out in outs:
        if not is_orca_output(out):
            continue
        d = out.parent
        job = _job_name(out)
        base = job or out.stem
        if job and job != out.stem and (d / f"{out.stem}.gbw").exists():
            problems.append((out, f"output is named {out.stem!r} but the job name is {job!r}, and both "
                                  f"{out.stem}.gbw and {job}.gbw candidates exist; ambiguous"))
            continue
        gbw = d / f"{base}.gbw"
        if not gbw.exists():
            others = sorted(p.name for p in d.glob("*.gbw"))
            hint = f" (other .gbw here, NOT associated: {', '.join(others)})" if others else ""
            problems.append((out, f"no {base}.gbw next to the output{hint}"))
            continue

        def opt(ext):
            p = d / f"{base}{ext}"
            return p if p.exists() else None

        calcs.append(Calculation(
            out_path=out, directory=d, base=base, gbw=gbw,
            uno=opt(".uno"), uco=opt(".uco"), property_txt=opt(".property.txt"),
            existing_cubes=sorted(d.glob(f"{base}*.cube")),
        ))

    # two outputs claiming the same orbital file -> both ambiguous
    by_gbw = {}
    for c in calcs:
        by_gbw.setdefault(c.gbw, []).append(c)
    unique = []
    for gbw, cs in by_gbw.items():
        if len(cs) > 1:
            for c in cs:
                problems.append((c.out_path, f"{gbw.name} is claimed by {len(cs)} outputs "
                                             f"({', '.join(x.out_path.name for x in cs)}); ambiguous"))
        else:
            unique.append(cs[0])

    # output folder: <dir>/<output_dir_name> (or <root>/<output_dir_name>/<rel dir> when central),
    # plus /<base> when a directory holds several jobs
    per_dir = {}
    for c in unique:
        per_dir.setdefault(c.directory, []).append(c)
    for d, cs in per_dir.items():
        for c in cs:
            base_dir = root / output_dir_name / d.relative_to(root) if central_output else d / output_dir_name
            c.output_dir = base_dir / (c.base if len(cs) > 1 else "")
    unique.sort(key=lambda c: str(c.out_path))
    return unique, problems
