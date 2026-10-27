"""
Positive / negative orbital isosurfaces from a cube.

Cube values are orbital *amplitudes* psi(r) (signed, bohr^-3/2), not densities:
the positive lobe is psi = +iso and the negative lobe is psi = -iso.

Marching cubes runs in grid-index space (unit spacing) and the vertices are
mapped to Cartesian coordinates with the full affine voxel transform
r = origin + ijk @ axes, so non-orthogonal grids are handled exactly
(linear interpolation along grid edges commutes with an affine map).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.measure import marching_cubes

from .cube import BOHR_TO_ANGSTROM


@dataclass
class Lobe:
    sign: int                 # +1 or -1
    vertices: np.ndarray      # (n, 3) Angstrom, in the cube's own frame
    faces: np.ndarray         # (m, 3) vertex indices


def _boundary_max(vol):
    return max(np.abs(vol[[0, -1], :, :]).max(), np.abs(vol[:, [0, -1], :]).max(), np.abs(vol[:, :, [0, -1]]).max())


def extract_lobes(cube, isovalue, value_index=0, phase=1.0):
    """
    Returns ({'positive': Lobe | None, 'negative': Lobe | None}, notes, stats).

    `phase` (+1/-1) multiplies the volume before contouring; it is applied only
    when the caller explicitly asks for a phase convention.
    """
    if isovalue <= 0:
        raise ValueError("isovalue must be a positive amplitude; both signs are contoured")
    vol = phase * cube.volume(value_index)
    vmin, vmax = float(vol.min()), float(vol.max())
    notes = []
    lobes = {"positive": None, "negative": None}
    for name, sign in (("positive", 1), ("negative", -1)):
        field = sign * vol
        if field.max() <= isovalue:
            notes.append(f"no {name} lobe: max {name} amplitude {field.max():.3g} <= isovalue {isovalue:g}")
            continue
        verts, faces, _, _ = marching_cubes(field, level=isovalue, allow_degenerate=False)
        if len(faces) == 0:
            notes.append(f"no {name} lobe: marching cubes returned no triangles")
            continue
        xyz = cube.grid_to_cartesian(verts) * BOHR_TO_ANGSTROM
        lobes[name] = Lobe(sign, xyz, faces.astype(np.int64))
    if _boundary_max(vol) > isovalue:
        notes.append("isosurface reaches the cube boundary (surface clipped; enlarge the grid box)")
    stats = {"min_amplitude": vmin, "max_amplitude": vmax}
    return lobes, notes, stats


def phase_factor(cube, convention, value_index=0):
    """
    Explicit, documented sign convention for display only (the cube is not changed).
      'none'             -> +1 always (sign exactly as written by ORCA)
      'largest_positive' -> make the largest-|psi| grid value positive
    """
    if convention in (None, "none"):
        return 1.0
    if convention == "largest_positive":
        vol = cube.volume(value_index)
        return 1.0 if vol.max() >= -vol.min() else -1.0
    raise ValueError(f"unknown phase convention {convention!r}")
