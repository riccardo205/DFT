#!/usr/bin/env python3
"""
Single exchange convention for all ORCA analysis scripts (docs/Riccardo/exchange_couplings.tex):

    H = - sum_{i<j} J_ij S_i . S_j        (spin-1/2 sites, dimensionless spins)
    J > 0 : ferromagnetic (FM)            J < 0 : antiferromagnetic (AFM)

    DeltaE = E_HS - E_BS                  (always this order; DeltaE > 0 <=> BS lower)

Diradical (two sites), with the column names used in the CSV outputs:
    J_ideal_meV                  = -2 * DeltaE                                 (ideal broken-symmetry mapping)
    J_yamaguchi_meV              = -2 * DeltaE / (<S^2>_HS - <S^2>_BS)         (Yamaguchi projection)
    E_T_minus_E_S_yamaguchi_meV  = -J_yamaguchi_meV                            (triplet - singlet gap)

N sites from collinear broken-symmetry energies (zero field):
    E_k = E_0 - 1/4 sum_{i<j} J_ij sigma_i^(k) sigma_j^(k),   sigma_i = +1 (up) / -1 (down)
fit_heisenberg() solves this least-squares problem for E_0 and the PHYSICAL J_ij directly
(design-matrix coefficient -1/4 sigma_i sigma_j); nothing is rescaled afterwards.

Pair labels: sites are numbered 1..N in the order of the configuration string, so for the dimers
(radicals A B C D) J_12 = J_AB, J_13 = J_AC, J_14 = J_AD, J_23 = J_BC, J_24 = J_BD, J_34 = J_CD.

Configuration checks (kept separate):
    pattern_ok        signs of the fragment spins equal the requested u/d pattern; for M_S = 0 the
                      global reversal (all spins flipped, same energy at zero field) is accepted too
    local_moments_ok  every |fragment spin| >= MIN_LOCAL_MOMENT (a full local S = 1/2 moment)
Only configurations with both checks true may enter fit_heisenberg().
"""

from itertools import combinations

import numpy as np

HARTREE_TO_MEV = 27211.386245988
MIN_LOCAL_MOMENT = 0.5


# ---------------------------------------------------------------- diradicals
def delta_e(e_hs, e_bs):
    """DeltaE = E_HS - E_BS (same units as the inputs)."""
    return e_hs - e_bs


def j_ideal(delta_e_value):
    """J_ideal = -2 * DeltaE, DeltaE = E_HS - E_BS."""
    return -2.0 * delta_e_value


def j_yamaguchi(delta_e_value, s2_hs, s2_bs):
    """J = -2 * DeltaE / (<S^2>_HS - <S^2>_BS), DeltaE = E_HS - E_BS."""
    return -2.0 * delta_e_value / (s2_hs - s2_bs)


def gap_t_minus_s(j):
    """E_T - E_S = -J."""
    return -j


# ---------------------------------------------------------------- collinear configurations
def sigmas(config):
    """'uudd' -> [+1, +1, -1, -1]."""
    if set(config) - set("ud"):
        raise ValueError(f"configuration {config!r} must consist of u/d")
    return [1 if s == "u" else -1 for s in config]


def pair_labels(n_sites):
    """['12', '13', ...] for sites numbered 1..n."""
    return [f"{i + 1}{j + 1}" for i, j in combinations(range(n_sites), 2)]


def pattern_ok(config, fragment_spins):
    """Signs of fragment_spins (sequence, site order) match the u/d pattern, or, for M_S = 0, its
    global reversal (equivalent at zero field)."""
    s = sigmas(config)
    signs = [int(np.sign(x)) for x in fragment_spins]
    if signs == s:
        return True
    return sum(s) == 0 and signs == [-x for x in s]


def local_moments_ok(fragment_spins, min_moment=MIN_LOCAL_MOMENT):
    """Every |fragment spin| >= min_moment."""
    return all(abs(x) >= min_moment for x in fragment_spins)


# ---------------------------------------------------------------- fit
def design_matrix(configs):
    """Rows [1, -1/4 sigma_i sigma_j for every pair i<j] for each configuration."""
    n = len(configs[0])
    rows = []
    for c in configs:
        s = sigmas(c)
        rows.append([1.0] + [-0.25 * s[i] * s[j] for i, j in combinations(range(n), 2)])
    return np.array(rows)


def fit_heisenberg(energies_mev):
    """
    Least-squares fit of E_k = E_0 - 1/4 sum_{i<j} J_ij sigma_i sigma_j.

    energies_mev: {config: energy in meV}, configs as u/d strings of equal length N.
    Returns {"J": {"12": J_12, ...} (meV), "E0": meV, "rank", "n_configs", "n_unknowns",
             "quality": "overdetermined" | "exact" | "insufficient",
             "residuals": {config: meV}, "rms": meV, "residual_norm": meV}.
    J is NaN (quality "insufficient") when the configurations do not determine all J_ij.
    """
    configs = list(energies_mev)
    n = len(configs[0]) if configs else 0
    labels = pair_labels(n) if n else []
    out = {"J": {p: np.nan for p in labels}, "E0": np.nan, "rank": 0, "n_configs": len(configs),
           "n_unknowns": 1 + len(labels), "quality": "insufficient", "residuals": {},
           "rms": np.nan, "residual_norm": np.nan}
    if not configs:
        return out
    X = design_matrix(configs)
    out["rank"] = int(np.linalg.matrix_rank(X))
    if out["rank"] < X.shape[1]:
        return out
    e = np.array([energies_mev[c] for c in configs], dtype=float)
    coef, *_ = np.linalg.lstsq(X, e, rcond=None)
    res = e - X @ coef
    out.update({"J": dict(zip(labels, coef[1:])), "E0": float(coef[0]),
                "quality": "overdetermined" if len(configs) > X.shape[1] else "exact",
                "residuals": dict(zip(configs, res)),
                "rms": float(np.sqrt(np.mean(res ** 2))), "residual_norm": float(np.sqrt(np.sum(res ** 2)))})
    return out


def collinear_energy(config, j, e0=0.0):
    """E_k = E_0 - 1/4 sum J_ij sigma_i sigma_j for J given as {'12': ..., ...} (the forward model)."""
    s = sigmas(config)
    return e0 - 0.25 * sum(j[f"{i + 1}{k + 1}"] * s[i] * s[k] for i, k in combinations(range(len(s)), 2))
