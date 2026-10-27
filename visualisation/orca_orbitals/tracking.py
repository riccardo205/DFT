"""
Orbital INDEX vs orbital IDENTITY.

Rows are currently exported by energy/occupation order only: "alpha HOMO" means
"highest occupied alpha orbital of THIS calculation". Across a field sweep or a
torsion scan two orbitals can swap order, so equal labels do not guarantee the
same physical orbital.

This module is the single place where identity is assigned, so a tracking method
can be added without touching discovery, cube generation or rendering. A natural
implementation for this pipeline: after alignment, evaluate overlaps
S_ij = sum_r psi_i^A(r) psi_j^B(r) dV on a common grid (or with the AO overlap
matrix from the .gbw), match with the Hungarian algorithm on |S_ij|, and use
sign(S_ij) as an explicit, recorded phase convention. |S| is sign-independent,
so an arbitrary sign flip is never mistaken for a change of orbital.
"""

METHODS = ("energy_order",)


def assign_identity(rows, method="energy_order"):
    """Fill 'identity' and 'identity_method' for each metadata row in place."""
    if method not in METHODS:
        raise NotImplementedError(f"orbital tracking method {method!r} is not implemented (available: {METHODS})")
    for r in rows:
        spin = r.get("spin") or "-"
        r["identity"] = f"{r['orbital_type']}:{spin}:{r.get('orbital_label', '')}"
        r["identity_method"] = method
    return rows
