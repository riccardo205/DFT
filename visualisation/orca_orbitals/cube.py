"""
Gaussian cube reader / writer.

File layout
-----------
    line 1-2   free comments (ORCA: "Molecular orbital 177 of operator 0")
    line 3     NAtoms  Ox Oy Oz  [NVal]
    line 4-6   N_k  Vx Vy Vz          voxel vector of grid axis k
    NAtoms     Z  charge  x  y  z     one line per atom
    if NAtoms < 0:  NMO  id_1 ... id_NMO   (record may wrap over several lines)
    data       for i1: for i2: for i3: [one value per orbital / NVal]

Units: Gaussian convention -- lengths are bohr when N_k > 0 and angstrom when
N_k < 0 (all three counts must carry the same sign). Everything is converted
to bohr on reading.

Ordering: the last grid index runs fastest, i.e. the values map onto a numpy
C-ordered array of shape (N1, N2, N3[, nval]), and the Cartesian position of
voxel (i, j, k) is  origin + i*V1 + j*V2 + k*V3  (V_k need not be orthogonal).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903
ORCA_MO_COMMENT_RE = re.compile(r"Molecular orbital\s+(\d+)\s+of operator\s+(\d+)")


class CubeFormatError(ValueError):
    pass


@dataclass
class Cube:
    comments: tuple
    origin: np.ndarray          # (3,) bohr
    axes: np.ndarray            # (3, 3) bohr, row k = step along grid index k
    atomic_numbers: np.ndarray  # (n,) int
    nuclear_charges: np.ndarray  # (n,)
    positions: np.ndarray       # (n, 3) bohr
    data: np.ndarray            # (N1, N2, N3) or (N1, N2, N3, nval)
    orbital_ids: tuple = field(default_factory=tuple)

    @property
    def shape(self):
        return self.data.shape[:3]

    @property
    def n_values(self):
        return 1 if self.data.ndim == 3 else self.data.shape[3]

    def volume(self, k=0):
        """3-D array of the k-th value per voxel (k-th orbital in multi-MO cubes)."""
        if self.data.ndim == 3:
            if k != 0:
                raise IndexError("cube holds a single value per voxel")
            return self.data
        return self.data[..., k]

    def grid_to_cartesian(self, ijk):
        """Fractional/integer grid indices (..., 3) -> Cartesian bohr (..., 3)."""
        return self.origin + np.asarray(ijk, dtype=float) @ self.axes

    def cartesian_grid(self):
        n1, n2, n3 = self.shape
        idx = np.stack(np.meshgrid(np.arange(n1), np.arange(n2), np.arange(n3), indexing="ij"), axis=-1)
        return self.grid_to_cartesian(idx)

    @property
    def positions_angstrom(self):
        return self.positions * BOHR_TO_ANGSTROM

    def is_orthogonal(self, tol=1e-8):
        g = self.axes @ self.axes.T
        return np.allclose(g - np.diag(np.diag(g)), 0.0, atol=tol)

    def orca_mo(self):
        """(mo, operator) from an ORCA comment line, else None."""
        for c in self.comments:
            m = ORCA_MO_COMMENT_RE.search(c)
            if m:
                return int(m[1]), int(m[2])
        return None


def _floats(tokens, where):
    try:
        return [float(t.replace("D", "E").replace("d", "e")) for t in tokens]
    except ValueError as e:
        raise CubeFormatError(f"{where}: non-numeric value ({e})") from None


def _int(token, where):
    try:
        v = float(token)
    except ValueError:
        raise CubeFormatError(f"{where}: expected an integer, got {token!r}") from None
    if v != int(v):
        raise CubeFormatError(f"{where}: expected an integer, got {token!r}")
    return int(v)


def read_cube_header(path):
    """Only the comment lines and grid counts (cheap validation of existing cubes)."""
    with open(path) as fh:
        head = [fh.readline() for _ in range(6)]
    if len(head) < 6 or not head[5]:
        raise CubeFormatError(f"{path}: truncated header")
    counts = tuple(abs(_int(head[3 + k].split()[0], f"{path} line {4 + k}")) for k in range(3))
    return (head[0].rstrip("\n"), head[1].rstrip("\n")), counts


def read_cube(path):
    path = Path(path)
    text = path.read_text()
    if not text.strip():
        raise CubeFormatError(f"{path}: empty file")
    lines = text.split("\n")
    if len(lines) < 7:
        raise CubeFormatError(f"{path}: file too short for a cube header")
    comments = (lines[0].rstrip(), lines[1].rstrip())

    t = lines[2].split()
    if len(t) < 4:
        raise CubeFormatError(f"{path} line 3: expected 'NAtoms Ox Oy Oz [NVal]'")
    natoms_signed = _int(t[0], f"{path} line 3")
    origin = np.array(_floats(t[1:4], f"{path} line 3"))
    nval = _int(t[4], f"{path} line 3") if len(t) >= 5 else 1
    if nval < 1:
        raise CubeFormatError(f"{path} line 3: NVal must be >= 1")

    counts, axes = [], []
    for k in range(3):
        tk = lines[3 + k].split()
        if len(tk) < 4:
            raise CubeFormatError(f"{path} line {4 + k}: expected 'N Vx Vy Vz'")
        n = _int(tk[0], f"{path} line {4 + k}")
        if n == 0:
            raise CubeFormatError(f"{path} line {4 + k}: zero grid points")
        counts.append(n)
        axes.append(_floats(tk[1:4], f"{path} line {4 + k}"))
    signs = {np.sign(n) for n in counts}
    if len(signs) != 1:
        raise CubeFormatError(f"{path}: mixed signs in voxel counts {counts} (ambiguous units)")
    angstrom = counts[0] < 0
    counts = [abs(n) for n in counts]
    axes = np.array(axes)
    if abs(np.linalg.det(axes)) < 1e-12:
        raise CubeFormatError(f"{path}: voxel vectors are linearly dependent")

    natoms = abs(natoms_signed)
    pos = 6
    if len(lines) < pos + natoms:
        raise CubeFormatError(f"{path}: file ends inside the atom block")
    Z, charges, xyz = [], [], []
    for a in range(natoms):
        ta = lines[pos + a].split()
        if len(ta) != 5:
            raise CubeFormatError(f"{path} line {pos + a + 1}: atom line needs 5 fields, got {len(ta)}")
        Z.append(_int(ta[0], f"{path} line {pos + a + 1}"))
        vals = _floats(ta[1:], f"{path} line {pos + a + 1}")
        charges.append(vals[0])
        xyz.append(vals[1:])
    pos += natoms

    orbital_ids = ()
    if natoms_signed < 0:
        # orbital record: NMO followed by NMO ids, possibly wrapped over several lines
        rec, start = [], pos
        while pos < len(lines):
            rec.extend(lines[pos].split())
            pos += 1
            if rec and len(rec) >= 1 + _int(rec[0], f"{path} line {start + 1}"):
                break
        if not rec:
            raise CubeFormatError(f"{path}: missing orbital record after negative atom count")
        nmo = _int(rec[0], f"{path} line {start + 1}")
        if nmo < 1 or len(rec) != 1 + nmo:
            raise CubeFormatError(f"{path} line {start + 1}: orbital record has {len(rec) - 1} ids, header says {nmo}")
        orbital_ids = tuple(_int(v, f"{path} orbital record") for v in rec[1:])
        if nval not in (1, nmo):
            raise CubeFormatError(f"{path}: NVal={nval} conflicts with {nmo} orbitals in the orbital record")
        nval = nmo

    body = " ".join(lines[pos:])
    if "D" in body or "d" in body:
        body = body.replace("D", "E").replace("d", "e")
    try:
        values = np.array(body.split(), dtype=float)
    except ValueError as e:
        raise CubeFormatError(f"{path}: non-numeric volumetric data ({e})") from None
    expected = counts[0] * counts[1] * counts[2] * nval
    if values.size != expected:
        raise CubeFormatError(
            f"{path}: found {values.size} volumetric values, expected {expected} "
            f"({counts[0]}x{counts[1]}x{counts[2]} grid x {nval} value(s) per point)")
    if not np.all(np.isfinite(values)):
        raise CubeFormatError(f"{path}: volumetric data contains NaN/inf")
    data = values.reshape(*counts, nval) if nval > 1 else values.reshape(*counts)

    scale = 1.0 / BOHR_TO_ANGSTROM if angstrom else 1.0
    return Cube(
        comments=comments,
        origin=origin * scale,
        axes=axes * scale,
        atomic_numbers=np.array(Z, dtype=int),
        nuclear_charges=np.array(charges),
        positions=np.array(xyz, dtype=float).reshape(natoms, 3) * scale,
        data=data,
        orbital_ids=orbital_ids,
    )


def write_cube(path, cube, angstrom=False, ids_per_line=10):
    """Write a cube (used by the tests to build synthetic files)."""
    s = BOHR_TO_ANGSTROM if angstrom else 1.0
    sign = -1 if angstrom else 1
    natoms = len(cube.atomic_numbers)
    out = [cube.comments[0], cube.comments[1]]
    nat_field = -natoms if cube.orbital_ids else natoms
    o = cube.origin * s
    out.append(f"{nat_field:5d} {o[0]:12.6f} {o[1]:12.6f} {o[2]:12.6f}")
    for n, v in zip(cube.shape, cube.axes * s):
        out.append(f"{sign * n:5d} {v[0]:12.6f} {v[1]:12.6f} {v[2]:12.6f}")
    for z, q, p in zip(cube.atomic_numbers, cube.nuclear_charges, cube.positions * s):
        out.append(f"{z:5d} {q:12.6f} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}")
    if cube.orbital_ids:
        rec = [len(cube.orbital_ids), *cube.orbital_ids]
        for i in range(0, len(rec), ids_per_line):
            out.append(" ".join(f"{v:5d}" for v in rec[i:i + ids_per_line]))
    flat = cube.data.reshape(-1)
    n3v = cube.shape[2] * cube.n_values
    for row in flat.reshape(-1, n3v):        # Gaussian writes 6 values per line, restarting per (i1, i2)
        for i in range(0, row.size, 6):
            out.append(" ".join(f"{v:13.5E}" for v in row[i:i + 6]))
    Path(path).write_text("\n".join(out) + "\n")
