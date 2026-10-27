"""
Rigid-body alignment and the common display frame.

Every calculation gets one RigidTransform  x -> x @ R.T + t  (R proper, det = +1)
that is applied identically to its atoms and to its isosurface vertices, so the
orbital can never move relative to the molecule.

    display = frame( kabsch(calc -> reference) )

`frame` is fixed once from the reference geometry (principal axes, optional
user rotation) and the camera always looks down -z of the display frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RigidTransform:
    rotation: np.ndarray       # (3, 3), proper rotation
    translation: np.ndarray    # (3,)

    @staticmethod
    def identity():
        return RigidTransform(np.eye(3), np.zeros(3))

    def apply(self, x):
        return np.asarray(x, dtype=float) @ self.rotation.T + self.translation

    def then(self, other):
        """Transform that applies self first, then `other`."""
        return RigidTransform(other.rotation @ self.rotation, other.rotation @ self.translation + other.translation)

    def to_dict(self):
        return {"rotation": self.rotation.tolist(), "translation": self.translation.tolist()}

    @staticmethod
    def from_dict(d):
        return RigidTransform(np.array(d["rotation"], dtype=float), np.array(d["translation"], dtype=float))


def kabsch(P, Q):
    """
    Proper rotation R and translation t minimising sum |R p_i + t - q_i|^2.

    A reflection is never returned: if the optimal orthogonal matrix has det = -1
    the smallest singular direction is flipped (Kabsch 1978), so the handedness of
    the molecule is preserved.
    """
    P, Q = np.asarray(P, dtype=float), np.asarray(Q, dtype=float)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"kabsch needs two (n, 3) arrays of equal shape, got {P.shape} and {Q.shape}")
    if len(P) < 3:
        raise ValueError("at least 3 atoms are required for a rigid alignment")
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d if d != 0 else 1.0])
    R = Vt.T @ D @ U.T
    if S[1] < 1e-8 * max(S[0], 1e-12):
        raise ValueError("alignment atoms are (nearly) collinear; rotation is undefined")
    return RigidTransform(R, qc - R @ pc)


def rmsd(A, B):
    return float(np.sqrt(np.mean(np.sum((np.asarray(A) - np.asarray(B)) ** 2, axis=1))))


def _rotation_xyz(deg):
    rx, ry, rz = np.radians(deg)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def principal_frame(coords, extra_rotation_deg=(0.0, 0.0, 0.0)):
    """
    Transform taking `coords` to a frame centred on their centroid with
      x = longest in-plane extent, y = second, z = smallest extent (view axis),
    so looking down -z shows the largest projected area of the molecule.
    Eigenvector signs are fixed deterministically (positive third moment), and
    z = x cross y keeps the frame right-handed. `extra_rotation_deg` (about the
    display x, y, z axes, applied in that order) lets the user tilt the view.
    """
    X = np.asarray(coords, dtype=float)
    c = X.mean(axis=0)
    Y = X - c
    w, V = np.linalg.eigh(Y.T @ Y)          # ascending variance
    ex, ey = V[:, 2], V[:, 1]
    for k, e in enumerate((ex, ey)):
        proj = Y @ e
        m3 = np.sum(proj ** 3)
        s = np.sign(m3) if abs(m3) > 1e-6 * max(np.sum(np.abs(proj) ** 3), 1e-12) else np.sign(proj[np.argmax(np.abs(proj))])
        if s < 0:
            if k == 0:
                ex = -ex
            else:
                ey = -ey
    ez = np.cross(ex, ey)
    R = _rotation_xyz(extra_rotation_deg) @ np.vstack([ex, ey, ez])
    return RigidTransform(R, -R @ c)


def check_alignment_atoms(symbols, ref_symbols, atoms):
    """Raise ValueError if the alignment atoms are missing or have different elements."""
    n = len(symbols)
    bad = [i for i in atoms if i >= n]
    if bad:
        raise ValueError(f"alignment atoms {bad} do not exist (molecule has {n} atoms)")
    diff = [(i, symbols[i], ref_symbols[i]) for i in atoms if symbols[i] != ref_symbols[i]]
    if diff:
        raise ValueError(f"atom order differs from the reference at alignment atoms {diff[:5]}")
