"""Camera and image settings are identical across orbitals, regardless of lobe size."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from orca_orbitals.render import RenderStyle, camera_contains, frame_camera, molecule_meshes, render_orbital
from orca_orbitals.isosurface import extract_lobes
from orca_orbitals.tests.test_geometry import p_x, s_func, synthetic


class Rendering(unittest.TestCase):
    def test_same_camera_for_small_and_large_lobes(self):
        style = RenderStyle(window_size=(400, 300), anti_aliasing=None)
        symbols = ["C", "H", "H", "H"]
        coords = np.array([[0, 0, 0], [1.08, 0, 0], [-0.54, 0.93, 0], [-0.54, -0.93, 0]])
        cam = frame_camera(coords, 3.0, style.window_size)
        mol = molecule_meshes(symbols, coords, style)
        small, _, _ = extract_lobes(synthetic(s_func, np.zeros(3), n=30), 0.2)
        large, _, _ = extract_lobes(synthetic(p_x, np.zeros(3), n=40), 0.02)
        out = Path(tempfile.mkdtemp())
        states, images = [], []
        for name, lobes in (("small", small), ("large", large)):
            disp = {k: (v.vertices, v.faces) if v is not None else None for k, v in lobes.items()}
            states.append(render_orbital(out / f"{name}.png", mol, disp, cam, style))
            images.append(np.asarray(Image.open(out / f"{name}.png").convert("RGB")))
        self.assertEqual(states[0], states[1])
        s = states[0]
        self.assertTrue(s["parallel_projection"])
        np.testing.assert_allclose(s["position"], cam.position)
        np.testing.assert_allclose(s["focal_point"], cam.focal_point)
        np.testing.assert_allclose(s["view_up"], cam.view_up)
        self.assertAlmostEqual(s["parallel_scale"], cam.parallel_scale)
        self.assertEqual(images[0].shape, images[1].shape)
        self.assertEqual(images[0].shape[:2], (300, 400))
        for img in images:
            self.assertLess(img.mean(), 254)          # not a blank white image
        # rendering is deterministic: the same content gives the same pixels
        disp = {k: (v.vertices, v.faces) if v is not None else None for k, v in small.items()}
        render_orbital(out / "again.png", mol, disp, cam, style)
        np.testing.assert_array_equal(np.asarray(Image.open(out / "again.png").convert("RGB")), images[0])

    def test_framing_contains_atoms(self):
        pts = np.random.default_rng(0).normal(size=(30, 3)) * 4
        cam = frame_camera(pts, 2.0, (1600, 1200))
        self.assertTrue(camera_contains(cam, pts, (1600, 1200)))
        self.assertFalse(camera_contains(cam, pts * 3, (1600, 1200)))


if __name__ == "__main__":
    unittest.main()
