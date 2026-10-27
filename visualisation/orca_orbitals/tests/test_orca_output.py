"""HOMO/LUMO selection from ORCA outputs (synthetic text + real fixtures)."""

import tempfile
import unittest
from pathlib import Path

from orca_orbitals.pipeline import select_uco
from orca_orbitals.orca_output import (OrcaOutputError, check_usable, parse_frontier_label, parse_output,
                                       parse_uno_occupations, select_frontier)

DATA = Path(__file__).parent / "data" / "bs_methyl"
REAL_HS = Path(__file__).resolve().parents[3] / "runs/orca/short-2OS/angle_scans/visualization_orbitals/M1/high_spin/F0/high_spin.out"
LABELS = ["HOMO-1", "HOMO", "LUMO", "LUMO+1"]


def _table(occs, e0=-10.0):
    rows = []
    for i, n in enumerate(occs):
        e = e0 + i
        rows.append(f"{i:4d}   {n:.4f}     {e / 27.2114:10.6f}     {e:10.4f} ")
    return "\n".join(rows)


def fake_output(alpha, beta=None, converged=True, first_section=True, terminated=True):
    """Minimal ORCA-like output. An earlier, different orbital section precedes the final one."""
    parts = ["                                 * O   R   C   A *", "NAME = job.inp",
             " Multiplicity           Mult            ....    1",
             " Basis Dimension        Dim             ....   20"]
    if first_section:   # e.g. the HS step of a BS job: must be ignored
        parts += ["               *           SCF CONVERGED AFTER   5 CYCLES          *",
                  "----------------", "ORBITAL ENERGIES", "----------------",
                  "                 SPIN UP ORBITALS", "  NO   OCC          E(Eh)            E(eV) ",
                  _table([1] * 7 + [0] * 3), "", "                 SPIN DOWN ORBITALS",
                  "  NO   OCC          E(Eh)            E(eV) ", _table([1] * 3 + [0] * 7), ""]
    parts += ["CARTESIAN COORDINATES (ANGSTROEM)", "---------------------------------",
              "  C      0.000000    0.000000    0.000000", "  H      1.090000    0.000000    0.000000",
              "", ""]
    parts.append("               *           SCF CONVERGED AFTER  12 CYCLES          *" if converged
                 else "                  *           SCF NOT CONVERGED AFTER 125 CYCLES      *")
    na = sum(1 for n in alpha if n > 0.5)
    parts.append(f"N(Alpha)           :      {na - 3e-5:.12f} electrons")
    if beta is not None:
        parts.append(f"N(Beta)            :      {sum(1 for n in beta if n > 0.5) - 3e-5:.12f} electrons")
    parts += ["----------------", "ORBITAL ENERGIES", "----------------"]
    if beta is None:
        parts += ["", "  NO   OCC          E(Eh)            E(eV) ", _table(alpha), ""]
    else:
        parts += ["                 SPIN UP ORBITALS", "  NO   OCC          E(Eh)            E(eV) ", _table(alpha), "",
                  "                 SPIN DOWN ORBITALS", "  NO   OCC          E(Eh)            E(eV) ", _table(beta),
                  "*Only the first 10 virtual orbitals were printed.", ""]
    parts += ["********************************", "* MULLIKEN POPULATION ANALYSIS *"]
    if terminated:
        parts.append("                             ****ORCA TERMINATED NORMALLY****")
    return "\n".join(parts) + "\n"


class FrontierTests(unittest.TestCase):
    def parse(self, text):
        d = tempfile.mkdtemp()
        p = Path(d) / "job.out"
        p.write_text(text)
        return parse_output(p)

    def idx(self, rows):
        return {(r["spin"], r["label"]): r["index"] for r in rows}

    def test_high_spin_alpha_beta_differ(self):
        out = self.parse(fake_output([1] * 6 + [0] * 4, [1] * 4 + [0] * 6))
        check_usable(out)
        ix = self.idx(select_frontier(out, LABELS, ["alpha", "beta"]))
        self.assertEqual([ix["alpha", l] for l in LABELS], [4, 5, 6, 7])
        self.assertEqual([ix["beta", l] for l in LABELS], [2, 3, 4, 5])

    def test_last_section_is_used(self):
        out = self.parse(fake_output([1] * 5 + [0] * 5, [1] * 5 + [0] * 5))
        self.assertEqual(self.idx(select_frontier(out, ["HOMO"], ["alpha", "beta"])),
                         {("alpha", "HOMO"): 4, ("beta", "HOMO"): 4})

    def test_energies_and_occupations_stored(self):
        out = self.parse(fake_output([1] * 5 + [0] * 5, [1] * 5 + [0] * 5))
        r = [x for x in select_frontier(out, ["LUMO"], ["alpha"]) if x["spin"] == "alpha"][0]
        self.assertEqual(r["occupation"], 0.0)
        self.assertAlmostEqual(r["energy_ev"], -5.0, places=3)

    def test_restricted(self):
        out = self.parse(fake_output([2] * 4 + [0] * 6, first_section=False))
        self.assertEqual(self.idx(select_frontier(out, ["HOMO", "LUMO"], ["alpha", "beta"])),
                         {("closed", "HOMO"): 3, ("closed", "LUMO"): 4})

    def test_unconverged_rejected(self):
        out = self.parse(fake_output([1] * 5 + [0] * 5, [1] * 5 + [0] * 5, converged=False))
        with self.assertRaisesRegex(OrcaOutputError, "NOT converged"):
            check_usable(out)

    def test_abnormal_termination_rejected(self):
        out = self.parse(fake_output([1] * 5 + [0] * 5, [1] * 5 + [0] * 5, terminated=False))
        with self.assertRaisesRegex(OrcaOutputError, "terminate"):
            check_usable(out)

    def test_fractional_occupation_rejected(self):
        out = self.parse(fake_output([1] * 4 + [0.5, 0.5] + [0] * 4, [1] * 5 + [0] * 5))
        with self.assertRaisesRegex(OrcaOutputError, "fractional"):
            select_frontier(out, LABELS, ["alpha"])

    def test_non_aufbau_rejected(self):
        out = self.parse(fake_output([1, 1, 0, 1, 1] + [0] * 5, [1] * 4 + [0] * 6))
        with self.assertRaisesRegex(OrcaOutputError, "non-aufbau"):
            select_frontier(out, LABELS, ["alpha"])

    def test_missing_orbital_section(self):
        d = tempfile.mkdtemp()
        p = Path(d) / "x.out"
        p.write_text("* O   R   C   A *\nnothing here\n")
        with self.assertRaises(OrcaOutputError):
            parse_output(p)

    def test_label_parsing(self):
        self.assertEqual(parse_frontier_label("HOMO-2"), ("HOMO", -2))
        self.assertEqual(parse_frontier_label("lumo+1"), ("LUMO", 1))
        for bad in ("HOMO+1", "LUMO-1", "SOMO", "HOMO-"):
            with self.assertRaises(ValueError):
                parse_frontier_label(bad)

    def test_fixture_bs_methyl(self):
        out = parse_output(DATA / "bs_methyl.out")
        check_usable(out)
        self.assertEqual(out.electronic_state(), "BS (Ms=0)")
        ix = self.idx(select_frontier(out, LABELS, ["alpha", "beta"]))
        self.assertEqual(ix["alpha", "HOMO"], 8)     # 9 alpha electrons, zero-based
        self.assertEqual(ix["beta", "LUMO"], 9)
        self.assertAlmostEqual(out.uco_overlaps[8], 0.02489)
        occ = parse_uno_occupations(DATA / "bs_methyl.property.txt")
        self.assertEqual(len(occ), 58)
        self.assertAlmostEqual(occ[8], 1.0248881855, places=8)
        self.assertAlmostEqual(occ[9], 0.9751118145, places=8)

    def test_uno_occupations_from_out_skip_spin_table(self):
        text = """
-------------------------
UHF NATURAL SPIN ORBITALS
-------------------------

Occupation Numbers:
N[  0]=   0.99969 N[  1]=   0.05564 N[  2]=   0.05562 N[  3]=   0.02470

--------------------
UHF NATURAL ORBITALS
--------------------

Occupation Numbers:
N[  0]=   2.00000 N[  1]=   1.02489 N[  2]=   0.97511
N[  3]=   0.00155 N[  4]=   0.00000

QR-MO GENERATION
"""
        self.assertEqual(parse_uno_occupations(None, text), {0: 2.0, 1: 1.02489, 2: 0.97511, 3: 0.00155, 4: 0.0})
        self.assertIsNone(parse_uno_occupations(None, "no natural orbitals here"))

    def test_uco_selection_bs_and_hs(self):
        bs = {175: 0.99, 176: 0.99, 177: 0.0188}
        self.assertEqual(select_uco(bs, 178.0, 178.0, 0.95, 4), ([177], [177], []))
        # triplet: entry 177 pairs occupied alpha 177 with EMPTY beta 177 -> not a pair
        hs = {175: 0.98968, 176: 0.98903, 177: -0.0}
        self.assertEqual(select_uco(hs, 179.0, 177.0, 0.95, 4), ([], [], [177, 178]))

    @unittest.skipUnless(REAL_HS.exists(), "real short-2OS output not available")
    def test_real_triplet(self):
        out = parse_output(REAL_HS)
        check_usable(out)
        self.assertEqual(out.electronic_state(), "HS (Ms=1)")
        ix = self.idx(select_frontier(out, LABELS, ["alpha", "beta"]))
        self.assertEqual((ix["alpha", "HOMO"], ix["alpha", "LUMO"]), (178, 179))
        self.assertEqual((ix["beta", "HOMO"], ix["beta", "LUMO"]), (176, 177))


if __name__ == "__main__":
    unittest.main()
