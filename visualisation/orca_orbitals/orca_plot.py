"""
Non-interactive driver for `orca_plot <file> -i` (verified against ORCA 6.1.1).

Menu used (main prompt "Enter a number:"):
    2  Enter no of orbital to plot         -> "Enter MO:"    zero-based index
    3  Enter operator of orbital           -> "Enter OP:"    0 = alpha, 1 = beta
    4  Enter number of grid intervals      -> "Enter NGRID:" points per axis
    5  Select output file format           -> "Enter Format:" 7 = Gaussian cube
   11  Generate the plot                   -> writes <base>.mo<N><a|b>.cube
   12  exit this program
Menu 1 (plot type) is deliberately NOT used: in 6.1.1 it opens <base>.densitiesinfo
and aborts if that file is missing. MO-PLOT is the default and is verified from the
"PlotType ... MO-PLOT" line instead.

Safety: orca_plot writes <base>.xyz and the cubes next to the file it opens, so it is
always run inside a private work directory on a symlink to the original file.
Asking for an operator that does not exist prints an error and leaves an EMPTY cube,
so the operator count is read first and every cube header is validated.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cube import CubeFormatError, ORCA_MO_COMMENT_RE, read_cube_header

N_ORB_RE = re.compile(r"Number of available orbitals\s*:\s*(\d+)")
N_OP_RE = re.compile(r"Number of operators\s*:\s*(\d+)")
PLOT_TYPE_RE = re.compile(r"PlotType\s+\.+\s+(\S.*)")
OUTPUT_FILE_RE = re.compile(r"Output file:\s*(\S+)")
ERROR_RE = re.compile(r"FATAL ERROR|^\s*Error\b|ABORTING", re.M | re.I)
REQUIRED_MENU = (
    "2 - Enter no of orbital to plot",
    "3 - Enter operator of orbital",
    "4 - Enter number of grid intervals",
    "5 - Select output file format",
    "11 - Generate the plot",
    "12 - exit this program",
)
CUBE_FORMAT_LINE = re.compile(r"^\s*7\s*-\s*3D\s+Gaussian cube", re.M)


class OrcaPlotError(RuntimeError):
    def __init__(self, message, log=""):
        super().__init__(message)
        self.log = log


@dataclass
class OrbitalFileInfo:
    n_orbitals: int
    n_operators: int
    symbols: list
    coords: np.ndarray        # geometry stored in the orbital file, Angstrom


@dataclass
class CubeRequest:
    index: int
    operator: int
    dest: Path


def find_orca_plot(user_path=None):
    candidates = []
    if user_path:
        candidates.append(Path(user_path).expanduser())
    which = shutil.which("orca_plot")
    if which:
        candidates.append(Path(which))
    for env in ("ORCA_DIR", "ORCA_PATH", "ORCA_HOME"):
        if os.environ.get(env):
            candidates.append(Path(os.environ[env]) / "orca_plot")
    candidates += sorted(Path.home().glob("software/orca*/orca_plot"), reverse=True)
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c.resolve()
    raise FileNotFoundError("orca_plot not found: set ORCA_PLOT in the configuration or put it on PATH")


def _prepare_workdir(source, workdir):
    """Fresh work dir with a symlink named exactly like the source file."""
    source = Path(source).resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    link = workdir / source.name
    try:
        link.symlink_to(source)
    except OSError:
        shutil.copy2(source, link)
    return link


def _run(exe, link, script, timeout):
    try:
        p = subprocess.run([str(exe), link.name, "-i"], input=script, cwd=link.parent,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        log = (e.stdout or "") if isinstance(e.stdout, str) else (e.stdout or b"").decode(errors="replace")
        raise OrcaPlotError(f"orca_plot timed out after {timeout} s", f"STDIN:\n{script}\n\nSTDOUT:\n{log}") from None
    return p.returncode, f"COMMAND: {exe} {link.name} -i\nSTDIN:\n{script}\nRETURN CODE: {p.returncode}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}"


def _read_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    symbols, coords = [], []
    for line in lines[2:2 + n]:
        t = line.split()
        symbols.append(t[0])
        coords.append([float(v) for v in t[1:4]])
    return symbols, np.array(coords)


def probe(exe, source, workdir, timeout=120):
    """Open the orbital file, verify the menu, and read orbital/operator counts and geometry."""
    link = _prepare_workdir(source, workdir)
    try:
        rc, log = _run(exe, link, "12\n", timeout)
        missing = [m for m in REQUIRED_MENU if m not in log]
        if missing:
            raise OrcaPlotError(f"unexpected orca_plot menu (missing {missing}); interface not supported", log)
        pt = PLOT_TYPE_RE.search(log)
        if not pt or "MO-PLOT" not in pt.group(1):
            raise OrcaPlotError(f"default plot type is {pt.group(1) if pt else '?'}, expected MO-PLOT", log)
        nor, nop = N_ORB_RE.search(log), N_OP_RE.search(log)
        if not nor or not nop:
            raise OrcaPlotError("could not read orbital/operator counts from orca_plot", log)
        xyz = workdir / (link.stem + ".xyz")
        if not xyz.exists():
            raise OrcaPlotError("orca_plot did not write the geometry file", log)
        symbols, coords = _read_xyz(xyz)
        return OrbitalFileInfo(int(nor[1]), int(nop[1]), symbols, coords)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def validate_cube_file(path, index, operator, grid):
    """Raise CubeFormatError unless `path` is a non-empty cube of the requested MO/operator/grid."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise CubeFormatError(f"{path.name}: missing or empty")
    comments, counts = read_cube_header(path)
    m = ORCA_MO_COMMENT_RE.search(comments[1]) or ORCA_MO_COMMENT_RE.search(comments[0])
    if not m:
        raise CubeFormatError(f"{path.name}: not an ORCA molecular-orbital cube ({comments[1]!r})")
    if (int(m[1]), int(m[2])) != (index, operator):
        raise CubeFormatError(f"{path.name}: contains MO {m[1]} op {m[2]}, expected MO {index} op {operator}")
    if grid is not None and counts != (grid, grid, grid):
        raise CubeFormatError(f"{path.name}: grid {counts}, expected {grid}^3")


def generate_cubes(exe, source, requests, grid, workdir, info, timeout_per_cube=120):
    """
    Generate all requested cubes from one orbital file in a single orca_plot session.

    Returns {request_position: (ok: bool, message: str)} and the full terminal log.
    Successful cubes are moved to request.dest.
    """
    results = {}
    todo = []
    for k, r in enumerate(requests):
        if not 0 <= r.operator < info.n_operators:
            results[k] = (False, f"operator {r.operator} not in file ({info.n_operators} operator(s))")
        elif not 0 <= r.index < info.n_orbitals:
            results[k] = (False, f"orbital {r.index} not in file ({info.n_orbitals} orbitals)")
        else:
            todo.append(k)
    if not todo:
        return results, ""

    link = _prepare_workdir(source, workdir)
    script = f"4\n{int(grid)}\n5\n7\n"
    for k in todo:
        script += f"2\n{requests[k].index}\n3\n{requests[k].operator}\n11\n"
    script += "12\n"
    t0 = time.time()
    try:
        rc, log = _run(exe, link, script, timeout_per_cube * (len(todo) + 1))
        if not CUBE_FORMAT_LINE.search(log):
            raise OrcaPlotError("Gaussian-cube format (7) not offered by this orca_plot", log)
        written = OUTPUT_FILE_RE.findall(log)
        errors = ERROR_RE.findall(log)
        for k in todo:
            r = requests[k]
            name = f"{link.stem}.mo{r.index}{'ab'[r.operator]}.cube"
            produced = workdir / name
            try:
                if name not in written:
                    raise CubeFormatError(f"orca_plot did not report writing {name}")
                validate_cube_file(produced, r.index, r.operator, grid)
            except CubeFormatError as e:
                results[k] = (False, str(e) + (f" | orca_plot errors: {errors}" if errors else ""))
                continue
            r.dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(produced), str(r.dest))
            results[k] = (True, f"generated in {time.time() - t0:.1f} s (session)")
        return results, log
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
