#!/usr/bin/env python3
"""
Synthetic tests of visualisation/exchange_convention.py.
Run with  python -m pytest visualisation/tests   or   python visualisation/tests/test_exchange_convention.py
"""

import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exchange_convention as X  # noqa: E402

DIMER_CONFIGS = ["uuuu", "duuu", "uduu", "uudu", "uuud", "dduu", "dudu", "duud"]
J_TRUE = {"12": +0.49, "13": -0.77, "14": +0.24, "23": +0.46, "24": -0.32, "34": +0.96}   # meV
E0 = -112943.3 * 1000                                                                     # meV


def synthetic(configs, j=J_TRUE, e0=E0, four_spin=0.0):
    return {c: X.collinear_energy(c, j, e0) + four_spin * np.prod(X.sigmas(c)) for c in configs}


def test_four_spin_fit_recovers_j():
    fit = X.fit_heisenberg(synthetic(DIMER_CONFIGS))
    assert fit["quality"] == "overdetermined" and fit["rank"] == 7
    for p, v in J_TRUE.items():
        assert abs(fit["J"][p] - v) < 1e-6, (p, fit["J"][p], v)
    assert abs(fit["E0"] - E0) < 1e-6 and fit["rms"] < 1e-6


def test_any_seven_configurations_are_exact():
    for missing in DIMER_CONFIGS:
        fit = X.fit_heisenberg(synthetic([c for c in DIMER_CONFIGS if c != missing]))
        assert fit["quality"] == "exact", missing
        assert all(abs(fit["J"][p] - v) < 1e-6 for p, v in J_TRUE.items()), missing


def test_six_configurations_insufficient():
    fit = X.fit_heisenberg(synthetic(DIMER_CONFIGS[:6]))
    assert fit["quality"] == "insufficient" and all(np.isnan(v) for v in fit["J"].values())


def test_four_spin_term_leaves_j_unchanged():
    k = 0.3
    fit = X.fit_heisenberg(synthetic(DIMER_CONFIGS, four_spin=k))
    assert all(abs(fit["J"][p] - v) < 1e-6 for p, v in J_TRUE.items())
    assert np.allclose(np.abs(list(fit["residuals"].values())), k)


def test_against_exact_heisenberg_expectation_values():
    """Collinear <H> of H = -sum J S_i.S_j built from Pauli matrices equals the fitted model."""
    sz = np.diag([0.5, -0.5])
    sx = np.array([[0, 0.5], [0.5, 0]])
    sy = np.array([[0, -0.5j], [0.5j, 0]])

    def op(o, i, n=4):
        m = np.array([[1.0]])
        for k in range(n):
            m = np.kron(m, o if k == i else np.eye(2))
        return m

    h = sum(-J_TRUE[f"{i + 1}{j + 1}"] * sum(op(o, i) @ op(o, j) for o in (sx, sy, sz))
            for i, j in combinations(range(4), 2))
    energies = {}
    for c in DIMER_CONFIGS:
        idx = int("".join("0" if s == "u" else "1" for s in c), 2)
        energies[c] = float(np.real(h[idx, idx]))
    fit = X.fit_heisenberg(energies)
    assert all(abs(fit["J"][p] - v) < 1e-9 for p, v in J_TRUE.items())


def test_diradical_formulas():
    j = 1.7
    e_t, e_s = -j / 4, 3 * j / 4                     # eigenvalues of H = -J S1.S2
    e_hs, e_bs = -j / 4, +j / 4                      # collinear uu / ud
    d = X.delta_e(e_hs, e_bs)
    assert abs(X.j_ideal(d) - j) < 1e-12
    assert abs(X.j_yamaguchi(d, 2.0, 1.0) - j) < 1e-12
    assert abs(X.gap_t_minus_s(j) - (e_t - e_s)) < 1e-12


def test_pattern_and_moment_checks():
    assert X.pattern_ok("dduu", [-1, -1, 1, 1])
    assert X.pattern_ok("dduu", [1, 1, -1, -1])            # M_S = 0: global reversal accepted
    assert not X.pattern_ok("duuu", [1, -1, -1, -1])       # M_S = 1: reversal not equivalent
    assert not X.pattern_ok("dduu", [-1, 1, -1, 1])
    assert X.local_moments_ok([0.6, -0.9, 1.0, -1.0])
    assert not X.local_moments_ok([0.44, 1.55, 1.0, 1.0])


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
