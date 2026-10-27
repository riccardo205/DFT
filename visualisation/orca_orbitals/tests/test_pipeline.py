"""
End-to-end checks on a real ORCA 6.1.1 broken-symmetry fixture (two CH3 radicals
5 A apart, UKS PBE0/def2-SVP with ! UNO UCO). Needs orca_plot; skipped otherwise.

Physical validation: in the BS solution the alpha and beta magnetic orbitals
(HOMO, and UCO pair 8 with S = 0.025) must sit on DIFFERENT carbons, and every
lobe must be centred on a carbon, not displaced from the atoms.
"""

import hashlib
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from orca_orbitals.cube import read_cube
from orca_orbitals.discovery import discover
from orca_orbitals.orca_plot import find_orca_plot
from orca_orbitals.pipeline import Settings, run
from orca_orbitals.render import RenderStyle

DATA = Path(__file__).parent / "data" / "bs_methyl"
try:
    ORCA_PLOT = find_orca_plot()
except FileNotFoundError:
    ORCA_PLOT = None


def tree_hash(root):
    """Sorted per-file hashes of everything outside generated output folders."""
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and "orbital_analysis" not in p.parts:
            h.update(str(p.relative_to(root)).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def fixture_copy(exts=("inp", "out", "gbw", "uno", "uco", "property.txt"), sub="bs_methyl"):
    root = Path(tempfile.mkdtemp())
    d = root / sub
    d.mkdir(parents=True)
    for e in exts:
        shutil.copy2(DATA / f"bs_methyl.{e}", d / f"bs_methyl.{e}")
    return root, d


def settings(root, **kw):
    kw.setdefault("grid_size", 40)
    kw.setdefault("central_output", False)
    return Settings(root=root, align_atoms=None, style=RenderStyle(window_size=(480, 360), anti_aliasing=None), **kw)


def psi2_centroid(cube):
    w = cube.volume() ** 2
    g = cube.cartesian_grid()
    return (g * w[..., None]).sum(axis=(0, 1, 2)) / w.sum()


class Discovery(unittest.TestCase):
    def test_strict_matching(self):
        root, d = fixture_copy(("out", "gbw"))
        (d / "slurm.out").write_text("not an ORCA file\n")
        shutil.copy2(d / "bs_methyl.gbw", d / "guess.gbw")                 # decoy
        other = root / "no_gbw"
        other.mkdir()
        shutil.copy2(d / "bs_methyl.out", other / "bs_methyl.out")
        shutil.copy2(d / "bs_methyl.gbw", other / "high_spin_reference.gbw")
        (root / "bs_methyl" / "orbital_analysis").mkdir()
        shutil.copy2(d / "bs_methyl.out", root / "bs_methyl" / "orbital_analysis" / "old.out")  # ignored dir
        calcs, problems = discover(root)
        self.assertEqual([c.gbw.name for c in calcs], ["bs_methyl.gbw"])
        self.assertIsNone(calcs[0].uno)
        self.assertEqual(len(problems), 1)
        self.assertIn("high_spin_reference.gbw", problems[0][1])
        self.assertIn("NOT associated", problems[0][1])

    def test_job_name_from_output(self):
        root, d = fixture_copy(("out", "gbw"))
        (d / "bs_methyl.out").rename(d / "slurm-123.out")
        calcs, _ = discover(root)
        self.assertEqual(len(calcs), 1)
        self.assertEqual(calcs[0].base, "bs_methyl")


@unittest.skipIf(ORCA_PLOT is None, "orca_plot not available")
class EndToEnd(unittest.TestCase):
    def test_full_fixture(self):
        root, d = fixture_copy()
        before = tree_hash(root)
        rows, skipped = run(settings(root))
        self.assertEqual(skipped, [])
        self.assertEqual(tree_hash(root), before, "original calculation files were modified")
        self.assertFalse(list(d.glob("*.xyz")) + list(d.glob("*.cube")), "orca_plot wrote next to the originals")
        self.assertFalse(list(root.rglob("_orca_plot_work*")))

        df = pd.DataFrame(rows)
        self.assertTrue((df.status == "ok").all(), df[["orbital_type", "orbital_label", "status", "message"]])
        mo = df[df.orbital_type == "MO"]
        self.assertEqual(len(mo), 8)
        homo = mo[mo.orbital_label == "HOMO"].set_index("spin")
        self.assertEqual(int(homo.loc["alpha", "orbital_index"]), 8)
        self.assertEqual(int(homo.loc["beta", "orbital_index"]), 8)
        uno = df[df.orbital_type == "UNO"]
        self.assertEqual(sorted(uno.orbital_index.astype(int)), [8, 9])
        self.assertTrue(uno.spin.fillna("").eq("").all())
        self.assertFalse(uno.orbital_label.str.contains("HOMO|LUMO").any())
        uco = df[df.orbital_type == "UCO"]
        self.assertEqual(sorted(uco.spin), ["alpha", "beta"])
        self.assertTrue(np.allclose(uco.uco_overlap.astype(float), 0.02489))

        od = d / "orbital_analysis"
        for f in ("contact_sheets/frontier_orbitals.png", "contact_sheets/UNO.png", "contact_sheets/UCO.png",
                  "orbital_metadata.csv", "processing_log.txt", "view_and_alignment.json",
                  "images/alpha_HOMO_mo8.png", "images/UNO/UNO_8.png", "images/UCO/UCO_8_beta.png",
                  "cubes/beta_LUMO+1_mo10.cube"):
            self.assertTrue((od / f).exists(), f)
        self.assertTrue((root / "orbital_analysis" / "view_settings.json").exists())

        # orbitals in the right place relative to the atoms
        def nearest_carbon(cube_path):
            c = read_cube(cube_path)
            carbons = c.positions[c.atomic_numbers == 6]
            dist = np.linalg.norm(carbons - psi2_centroid(c), axis=1)
            self.assertLess(dist.min(), 0.5, f"{cube_path.name}: density centroid not on a carbon")
            return int(np.argmin(dist))

        a = nearest_carbon(od / "cubes/alpha_HOMO_mo8.cube")
        b = nearest_carbon(od / "cubes/beta_HOMO_mo8.cube")
        self.assertNotEqual(a, b, "BS magnetic orbitals should be on different radical centres")
        ua = nearest_carbon(od / "cubes/UCO/UCO_8_alpha.cube")
        ub = nearest_carbon(od / "cubes/UCO/UCO_8_beta.cube")
        self.assertNotEqual(ua, ub)

        # second run: cubes are reused, view is reused, results identical
        rows2, _ = run(settings(root))
        self.assertTrue(all("cube reused" in r["message"] for r in rows2 if r["status"] == "ok"))

    def test_uno_uco_absent(self):
        root, d = fixture_copy(("out", "gbw"))
        rows, skipped = run(settings(root))
        self.assertEqual(skipped, [])
        self.assertEqual({r["orbital_type"] for r in rows}, {"MO"})
        log = (d / "orbital_analysis" / "processing_log.txt").read_text()
        self.assertIn("UNO: no .uno file; stage skipped", log)
        self.assertIn("UCO: no .uco file; stage skipped", log)
        self.assertFalse((d / "orbital_analysis" / "contact_sheets" / "UNO.png").exists())

    def test_uno_without_occupations_is_reported(self):
        root, d = fixture_copy(("out", "gbw", "uno"))
        rows, skipped = run(settings(root, render_uco=False))
        self.assertEqual(skipped, [])
        uno = [r for r in rows if r["orbital_type"] == "UNO"]
        self.assertEqual([r["status"] for r in uno], ["skipped"])

    def test_mismatched_gbw_is_rejected(self):
        root, d = fixture_copy(("out", "gbw"))
        out = d / "bs_methyl.out"
        text = out.read_text()
        # move the final geometry: the .gbw no longer belongs to this output
        head, sep, tail = text.rpartition("CARTESIAN COORDINATES (ANGSTROEM)")
        tail = re.sub(r"(\n\s+C\s+)(-?\d+\.\d+)", lambda m: f"{m[1]}{float(m[2]) + 0.3:.6f}", tail, count=1)
        out.write_text(head + sep + tail)
        rows, skipped = run(settings(root))
        self.assertEqual(len(skipped), 1)
        self.assertIn("geometry differs", skipped[0]["reason"])
        self.assertFalse(list((d / "orbital_analysis").glob("images/*.png")))

    def test_central_output(self):
        root, d = fixture_copy(("out", "gbw"), sub="geom/bs")
        rows, skipped = run(settings(root, central_output=True))
        self.assertEqual(skipped, [])
        od = root / "orbital_analysis" / "geom" / "bs"
        self.assertTrue((od / "contact_sheets" / "frontier_orbitals.png").exists())
        self.assertTrue(all(r["image_path"].startswith(str(od)) for r in rows))
        self.assertEqual([p.name for p in d.iterdir() if p.is_dir()], [], "nothing may be written in the job dir")

    def test_smoke_test_processes_one(self):
        root, _ = fixture_copy(sub="a")
        _, d2 = fixture_copy(sub="b")
        shutil.copytree(d2, root / "b")
        rows, _ = run(settings(root, smoke_test=True, render_uno=False, render_uco=False))
        self.assertEqual({r["calculation"] for r in rows}, {"a/bs_methyl"})


if __name__ == "__main__":
    unittest.main()
