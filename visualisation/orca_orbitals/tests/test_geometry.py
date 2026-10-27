"""Isosurface lobes, their position relative to atoms, and rigid alignment."""

import unittest

import numpy as np

from orca_orbitals.alignment import (RigidTransform, check_alignment_atoms, kabsch, principal_frame, rmsd)
from orca_orbitals.cube import BOHR_TO_ANGSTROM, Cube
from orca_orbitals.isosurface import extract_lobes, phase_factor

AXES = np.array([[0.25, 0.00, 0.00], [0.06, 0.25, 0.00], [0.03, -0.05, 0.25]])   # non-orthogonal, bohr


def synthetic(func, center_bohr, n=48):
    origin = np.asarray(center_bohr) - (n - 1) / 2 * AXES.sum(axis=0)
    idx = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), axis=-1)
    xyz = origin + idx @ AXES
    data = func(xyz - center_bohr)
    return Cube(("t", "t"), origin, AXES, np.array([6]), np.array([6.0]), np.array([center_bohr], float), data)


def p_x(r):
    return r[..., 0] * np.exp(-np.sum(r ** 2, axis=-1) / 2.0)


def s_func(r):
    return 0.3 * np.exp(-np.sum(r ** 2, axis=-1) / 2.0)


def random_rotation(seed):
    q = np.random.default_rng(seed).normal(size=4)
    a, b, c, d = q / np.linalg.norm(q)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


class Lobes(unittest.TestCase):
    def test_px_has_opposite_lobes_along_x(self):
        center = np.array([1.0, -2.0, 3.0])
        cube = synthetic(p_x, center)
        lobes, notes, _ = extract_lobes(cube, 0.1)
        atom = cube.positions_angstrom[0]
        pos = lobes["positive"].vertices.mean(axis=0) - atom
        neg = lobes["negative"].vertices.mean(axis=0) - atom
        self.assertGreater(pos[0], 0.4)
        self.assertLess(neg[0], -0.4)
        np.testing.assert_allclose(pos, -neg, atol=0.05)              # mirror images through the atom
        np.testing.assert_allclose(pos[1:], 0, atol=0.05)
        self.assertEqual(notes, [])

    def test_s_lobe_centred_on_atom(self):
        cube = synthetic(s_func, np.array([2.0, 0.5, -1.0]))
        lobes, _, _ = extract_lobes(cube, 0.05)
        self.assertIsNone(lobes["negative"])
        np.testing.assert_allclose(lobes["positive"].vertices.mean(axis=0), cube.positions_angstrom[0], atol=0.03)
        # sphere radius: 0.3 exp(-r^2/2) = 0.05 -> r = sqrt(2 ln 6) bohr
        r = np.linalg.norm(lobes["positive"].vertices - cube.positions_angstrom[0], axis=1)
        self.assertAlmostEqual(r.mean(), np.sqrt(2 * np.log(6)) * BOHR_TO_ANGSTROM, delta=0.03)

    def test_rigid_transform_keeps_orbital_on_atom(self):
        cube = synthetic(p_x, np.array([0.5, 0.5, 0.5]))
        lobes, _, _ = extract_lobes(cube, 0.1)
        T = RigidTransform(random_rotation(3), np.array([4.0, -1.0, 2.0]))
        atom, pos = T.apply(cube.positions_angstrom)[0], T.apply(lobes["positive"].vertices).mean(axis=0)
        before = np.linalg.norm(lobes["positive"].vertices.mean(axis=0) - cube.positions_angstrom[0])
        self.assertAlmostEqual(np.linalg.norm(pos - atom), before, places=8)

    def test_isovalue_above_data(self):
        lobes, notes, stats = extract_lobes(synthetic(s_func, np.zeros(3), n=20), 0.5)
        self.assertIsNone(lobes["positive"])
        self.assertIsNone(lobes["negative"])
        self.assertTrue(any("no positive lobe" in n for n in notes))

    def test_clipped_surface_reported(self):
        lobes, notes, _ = extract_lobes(synthetic(s_func, np.zeros(3), n=12), 0.01)
        self.assertTrue(any("boundary" in n for n in notes))

    def test_phase_convention_explicit(self):
        cube = synthetic(lambda r: -s_func(r), np.zeros(3), n=16)
        self.assertEqual(phase_factor(cube, "none"), 1.0)
        self.assertEqual(phase_factor(cube, "largest_positive"), -1.0)
        lobes, _, _ = extract_lobes(cube, 0.05, phase=-1.0)
        self.assertIsNotNone(lobes["positive"])
        self.assertIsNone(lobes["negative"])
        with self.assertRaises(ValueError):
            extract_lobes(cube, -0.05)


class Alignment(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.P = rng.normal(size=(12, 3)) * 2.0         # generic, chiral point set

    def test_recovers_rotation(self):
        R, t = random_rotation(7), np.array([1.0, -3.0, 0.5])
        Q = self.P @ R.T + t
        T = kabsch(self.P, Q)
        np.testing.assert_allclose(T.rotation, R, atol=1e-10)
        self.assertLess(rmsd(T.apply(self.P), Q), 1e-10)
        self.assertAlmostEqual(np.linalg.det(T.rotation), 1.0)

    def test_never_reflects(self):
        Q = self.P * np.array([-1, 1, 1])               # mirror image
        T = kabsch(self.P, Q)
        self.assertAlmostEqual(np.linalg.det(T.rotation), 1.0)
        self.assertGreater(rmsd(T.apply(self.P), Q), 0.5)   # a reflection would give 0

    def test_collinear_rejected(self):
        line = np.outer(np.arange(5.0), [1, 0, 0])
        with self.assertRaises(ValueError):
            kabsch(line, line)

    def test_compose(self):
        A = RigidTransform(random_rotation(1), np.array([1.0, 2, 3]))
        B = RigidTransform(random_rotation(2), np.array([-1.0, 0, 4]))
        np.testing.assert_allclose(A.then(B).apply(self.P), B.apply(A.apply(self.P)))
        np.testing.assert_allclose(RigidTransform.from_dict(A.to_dict()).apply(self.P), A.apply(self.P))

    def test_principal_frame_proper_and_deterministic(self):
        F = principal_frame(self.P)
        self.assertAlmostEqual(np.linalg.det(F.rotation), 1.0)
        X = F.apply(self.P)
        np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-12)
        var = X.var(axis=0)
        self.assertTrue(var[0] >= var[1] >= var[2])
        # same frame for a rigidly moved copy (view does not depend on the input orientation)
        R = random_rotation(9)
        X2 = principal_frame(self.P @ R.T + 5).apply(self.P @ R.T + 5)
        np.testing.assert_allclose(X, X2, atol=1e-8)

    def test_alignment_atom_checks(self):
        with self.assertRaisesRegex(ValueError, "do not exist"):
            check_alignment_atoms(["C"] * 3, ["C"] * 3, [0, 5])
        with self.assertRaisesRegex(ValueError, "atom order"):
            check_alignment_atoms(["C", "N", "C"], ["C", "C", "C"], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
