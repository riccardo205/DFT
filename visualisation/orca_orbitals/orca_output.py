"""
Parse the parts of an ORCA text output (and .property.txt) needed to choose
which orbitals to plot.

Only the *final* wavefunction matters: broken-symmetry jobs contain an HS SCF
followed by the BS SCF, and optimisations contain one SCF per geometry step, so
every quantity below is taken from the last occurrence in the file.

ORCA orbital numbering is zero-based, and orca_plot uses the same numbering, so
indices are stored exactly as printed by ORCA (no +1/-1 conversions anywhere).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HARTREE_TO_EV = 27.211386245988

VERSION_RE = re.compile(r"Program Version\s+(\S+)")
NAME_RE = re.compile(r"^NAME\s*=\s*(\S+)", re.M)
MULT_RE = re.compile(r"Multiplicity\s+Mult\s+\.+\s+(\d+)")
CHARGE_RE = re.compile(r"Total Charge\s+Charge\s+\.+\s+(-?\d+)")
NEL_RE = re.compile(r"Number of Electrons\s+NEL\s+\.+\s+(\d+)")
BASIS_DIM_RE = re.compile(r"Basis Dimension\s+Dim\s+\.+\s+(\d+)")
N_ALPHA_RE = re.compile(r"N\(Alpha\)\s*:\s*(-?\d+\.\d+)")
N_BETA_RE = re.compile(r"N\(Beta\)\s*:\s*(-?\d+\.\d+)")
S2_RE = re.compile(r"Expectation value of <S\*\*2>\s*:\s*(-?\d+\.\d+)")
EFIELD_RE = re.compile(r"^\|\s*\d+>\s*EField\s+([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)", re.M | re.I)
FINAL_E_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
SCF_OK_RE = re.compile(r"SCF CONVERGED AFTER\s+\d+\s+CYCLES")
SCF_FAIL_RE = re.compile(r"SCF NOT CONVERGED|wavefunction IS NOT (?:FULLY )?CONVERGED", re.I)
TERMINATED_RE = re.compile(r"\*\*\*\*ORCA TERMINATED NORMALLY\*\*\*\*")
ORB_HEADER_RE = re.compile(r"^-+\s*\nORBITAL ENERGIES\s*\n-+\s*$", re.M)
ORB_ROW_RE = re.compile(r"^\s*(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
COORD_BLOCK_RE = re.compile(r"CARTESIAN COORDINATES \(ANGSTROEM\)\s*\n-+\s*\n((?:\s*[A-Za-z]{1,3}\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s*\n)+)")
UCO_HEADER_RE = re.compile(r"UHF CORRESPONDING ORBITALS")
UCO_ROW_RE = re.compile(r"^\s*(\d+):\s+(-?\d+\.\d+)\s*$")
UNO_OUT_ROW_RE = re.compile(r"N\[\s*(\d+)\]\s*=\s*(-?\d+\.\d+)")
UNO_OUT_HEADER_RE = re.compile(r"^-+\s*\nUHF NATURAL ORBITALS\s*\n-+\s*$", re.M)
FRONTIER_LABEL_RE = re.compile(r"^(HOMO|LUMO)(?:([+-])(\d+))?$")

# A spin channel is "integer occupied" if every occupation is within this of 0/1 (UKS) or 0/2 (RKS).
OCC_TOL = 1e-3


class OrcaOutputError(ValueError):
    """The output cannot be used for orbital selection (reason in the message)."""


@dataclass
class OrbitalLevel:
    index: int          # ORCA zero-based orbital number (same as orca_plot)
    occupation: float
    energy_eh: float
    energy_ev: float


@dataclass
class OrcaOutput:
    path: Path
    version: str | None
    input_name: str | None
    terminated_normally: bool
    scf_converged: bool | None
    multiplicity: int | None
    charge: int | None
    n_electrons: int | None
    basis_dimension: int | None
    n_alpha: float | None
    n_beta: float | None
    s2: float | None
    efield: tuple | None
    final_energy: float | None
    orbitals: dict = field(default_factory=dict)   # "alpha"/"beta" (UKS) or "closed" (RKS) -> [OrbitalLevel]
    symbols: list = field(default_factory=list)
    coords: np.ndarray | None = None               # final geometry, Angstrom
    uco_overlaps: dict | None = None               # pair index -> overlap

    @property
    def restricted(self):
        return "closed" in self.orbitals

    @property
    def spin_channels(self):
        return list(self.orbitals)

    def electronic_state(self):
        """
        Label inferred from the *final* wavefunction, not from the folder name or the
        input multiplicity (FlipSpin/FinalMs jobs declare Mult 3 but end as Ms = 0).
        """
        if self.restricted:
            return "closed-shell"
        if self.n_alpha is None or self.n_beta is None:
            return "unknown"
        ms = round(self.n_alpha - self.n_beta) / 2
        if ms > 0:
            return f"HS (Ms={ms:g})"
        if self.s2 is not None and self.s2 > 0.1:
            return "BS (Ms=0)"
        return "singlet (Ms=0, no spin contamination)"


def _last_float(regex, text, cast=float):
    m = regex.findall(text)
    return cast(m[-1]) if m else None


def _parse_orbital_section(text):
    """Return {channel: [OrbitalLevel]} from the LAST 'ORBITAL ENERGIES' section."""
    headers = list(ORB_HEADER_RE.finditer(text))
    if not headers:
        raise OrcaOutputError("no 'ORBITAL ENERGIES' section")
    lines = text[headers[-1].end():].splitlines()
    tables, current, rows_started = {}, None, False
    for line in lines[1:]:
        s = line.strip()
        if s.startswith("SPIN UP ORBITALS"):
            current, rows_started = "alpha", False
            continue
        if s.startswith("SPIN DOWN ORBITALS"):
            current, rows_started = "beta", False
            continue
        if s.startswith("NO") and "OCC" in s:
            if current is None:
                current = "closed"
            tables.setdefault(current, [])
            continue
        m = ORB_ROW_RE.match(line)
        if m and current is not None:
            idx, occ, eh, ev = int(m[1]), float(m[2]), float(m[3]), float(m[4])
            tables[current].append(OrbitalLevel(idx, occ, eh, ev))
            rows_started = True
            continue
        if not rows_started:
            continue
        # anything else after rows have started ends the current table
        if current in ("beta", "closed"):
            break
        if s.startswith("*Only the first"):
            continue
        if s == "":
            continue
        break
    if not tables or not any(tables.values()):
        raise OrcaOutputError("'ORBITAL ENERGIES' section has no orbital rows")
    if "closed" in tables and len(tables) > 1:
        raise OrcaOutputError("orbital section mixes restricted and unrestricted tables")
    if ("alpha" in tables) != ("beta" in tables):
        raise OrcaOutputError("unrestricted orbital section is missing the alpha or beta table")
    for ch, rows in tables.items():
        idx = [r.index for r in rows]
        if idx != list(range(len(idx))):
            raise OrcaOutputError(f"{ch} orbital table is not a contiguous 0..N-1 list")
    return tables


def _parse_final_geometry(text):
    blocks = COORD_BLOCK_RE.findall(text)
    if not blocks:
        return [], None
    symbols, coords = [], []
    for line in blocks[-1].strip().splitlines():
        t = line.split()
        symbols.append(t[0])
        coords.append([float(v) for v in t[1:4]])
    return symbols, np.array(coords)


def _scf_status(text):
    """Status of the last SCF that finished before the final orbital printout."""
    headers = list(ORB_HEADER_RE.finditer(text))
    end = headers[-1].start() if headers else len(text)
    ok = [m.start() for m in SCF_OK_RE.finditer(text, 0, end)]
    bad = [m.start() for m in SCF_FAIL_RE.finditer(text, 0, end)]
    if not ok and not bad:
        return None
    return (max(ok) if ok else -1) > (max(bad) if bad else -1)


def parse_uco_overlaps(text):
    """Pair index -> alpha/beta corresponding-orbital overlap (last UCO block), or None."""
    heads = list(UCO_HEADER_RE.finditer(text))
    if not heads:
        return None
    overlaps, started = {}, False
    for line in text[heads[-1].end():].splitlines():
        m = UCO_ROW_RE.match(line)
        if m:
            overlaps[int(m[1])] = float(m[2])
            started = True
        elif started:
            break
    return overlaps or None


def parse_output(path):
    """Parse an ORCA .out file. Raises OrcaOutputError only if no orbital data can be read."""
    path = Path(path)
    text = path.read_text(errors="replace")
    efield = EFIELD_RE.findall(text)
    out = OrcaOutput(
        path=path,
        version=_last_float(VERSION_RE, text, str),
        input_name=_last_float(NAME_RE, text, str),
        terminated_normally=bool(TERMINATED_RE.search(text)),
        scf_converged=_scf_status(text),
        multiplicity=_last_float(MULT_RE, text, int),
        charge=_last_float(CHARGE_RE, text, int),
        n_electrons=_last_float(NEL_RE, text, int),
        basis_dimension=_last_float(BASIS_DIM_RE, text, int),
        n_alpha=_last_float(N_ALPHA_RE, text),
        n_beta=_last_float(N_BETA_RE, text),
        s2=_last_float(S2_RE, text),
        efield=tuple(float(v) for v in efield[-1]) if efield else None,
        final_energy=_last_float(FINAL_E_RE, text),
    )
    out.orbitals = _parse_orbital_section(text)
    out.symbols, out.coords = _parse_final_geometry(text)
    out.uco_overlaps = parse_uco_overlaps(text)
    return out


def check_usable(out):
    """Raise OrcaOutputError if the final wavefunction should not be visualised."""
    if out.scf_converged is None:
        raise OrcaOutputError("no SCF convergence message found")
    if not out.scf_converged:
        raise OrcaOutputError("final SCF is NOT converged")
    if not out.terminated_normally:
        raise OrcaOutputError("ORCA did not terminate normally (orbitals may not be final)")
    if out.coords is None:
        raise OrcaOutputError("no final Cartesian geometry in output")


# ---------------------------------------------------------------------------
# frontier orbitals
# ---------------------------------------------------------------------------

def parse_frontier_label(label):
    """'HOMO-1' -> ('HOMO', -1), 'LUMO+1' -> ('LUMO', 1). HOMO+n / LUMO-n are rejected."""
    m = FRONTIER_LABEL_RE.match(label.strip().upper())
    if not m:
        raise ValueError(f"orbital label {label!r} is not of the form HOMO[-n] or LUMO[+n]")
    kind, sign, n = m[1], m[2], int(m[3]) if m[3] else 0
    offset = -n if sign == "-" else n
    if (kind == "HOMO" and offset > 0) or (kind == "LUMO" and offset < 0):
        raise ValueError(f"{label!r} is ambiguous; use HOMO-n for occupied and LUMO+n for virtual orbitals")
    return kind, offset


def occupied_count(levels, full_occ):
    """Number of occupied orbitals, requiring integer and aufbau (contiguous) occupation."""
    occ = np.array([lv.occupation for lv in levels])
    frac = (np.abs(occ) > OCC_TOL) & (np.abs(occ - full_occ) > OCC_TOL)
    if frac.any():
        bad = [levels[i].index for i in np.flatnonzero(frac)[:5]]
        raise OrcaOutputError(f"fractional occupations (orbitals {bad}); HOMO/LUMO undefined")
    filled = np.abs(occ - full_occ) <= OCC_TOL
    n = int(filled.sum())
    if not filled[:n].all():
        holes = [levels[i].index for i in np.flatnonzero(~filled[:n])[:5]]
        raise OrcaOutputError(f"non-aufbau occupation (empty orbitals {holes} below occupied ones)")
    if n == 0:
        raise OrcaOutputError("no occupied orbitals")
    return n


def select_frontier(out, labels, spins):
    """
    Map frontier labels to ORCA orbital indices, separately for every spin channel.

    Returns a list of dicts (one per spin/label) with index, occupation, energies and
    a 'note' for near-degeneracies. Labels outside the printed orbital range are
    reported with status 'unavailable' instead of being guessed.
    """
    chosen = []
    channels = ["closed"] if out.restricted else [s for s in spins if s in out.orbitals]
    full = 2.0 if out.restricted else 1.0
    electron_count = {"alpha": out.n_alpha, "beta": out.n_beta, "closed": None}
    for ch in channels:
        levels = out.orbitals[ch]
        nocc = occupied_count(levels, full)
        ref = electron_count[ch]
        if ref is not None and abs(round(ref) - nocc) != 0:
            raise OrcaOutputError(f"{ch}: {nocc} occupied orbitals but N({ch})={ref:.4f}")
        homo = nocc - 1
        for label in labels:
            kind, offset = parse_frontier_label(label)
            idx = homo + offset if kind == "HOMO" else homo + 1 + offset
            row = {"spin": ch, "label": label, "index": idx, "status": "ok", "note": ""}
            if idx < 0 or idx >= len(levels):
                row.update(status="unavailable", note=f"orbital {idx} not printed in output")
            else:
                lv = levels[idx]
                row.update(occupation=lv.occupation, energy_eh=lv.energy_eh, energy_ev=lv.energy_ev)
                near = [levels[j].index for j in (idx - 1, idx + 1)
                        if 0 <= j < len(levels) and abs(levels[j].energy_ev - lv.energy_ev) < 0.01]
                if near:
                    row["note"] = f"near-degenerate (<10 meV) with orbital(s) {near}; identity may swap"
            chosen.append(row)
    return chosen


# ---------------------------------------------------------------------------
# natural orbitals (.property.txt, ORCA >= 5) with a fallback for older text outputs
# ---------------------------------------------------------------------------

def parse_uno_occupations(property_path=None, out_text=None):
    """
    UNO occupation numbers {index: occupation}, or None if they cannot be found.

    ORCA 6 does not print UNO occupations in the .out file; they are stored in
    <base>.property.txt as the last  $Natural_Orbitals ... &OccUno [&Dim (N,1)]  array.
    """
    if property_path is not None and Path(property_path).exists():
        text = Path(property_path).read_text(errors="replace")
        blocks = re.findall(r"^\$Natural_Orbitals\s*$(.*?)^\$End", text, re.M | re.S)
        if blocks:
            block = blocks[-1]
            m = re.search(r"&OccUno\s*\[.*?&Dim\s*\((\d+)\s*,\s*1\)\].*?\n", block)
            if m:
                n = int(m[1])
                occ = {}
                for line in block[m.end():].splitlines():
                    t = line.split()
                    if len(t) == 2 and t[0].isdigit():
                        occ[int(t[0])] = float(t[1])
                        if len(occ) == n:
                            break
                    elif t and t[0].startswith("&"):
                        break
                if len(occ) == n:
                    return occ
    if out_text is not None:
        # Print[P_UNO_OccNum] 1: a "UHF NATURAL SPIN ORBITALS" table is printed first and
        # must not be mistaken for the UNO table; use only the last UNO block.
        heads = list(UNO_OUT_HEADER_RE.finditer(out_text))
        if heads:
            occ = {}
            for line in out_text[heads[-1].end():].splitlines():
                rows = UNO_OUT_ROW_RE.findall(line)
                if rows:
                    occ.update((int(i), float(v)) for i, v in rows)
                elif occ:
                    break
            if occ:
                return occ
    return None
