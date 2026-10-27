"""
Off-screen PyVista rendering with a camera that is fixed by the caller.

The camera is never derived from what is being drawn: position, focal point,
view-up, orthographic projection and parallel scale come from a Camera object
(computed once per comparison set and saved to JSON), so two orbitals of the
same molecule are framed identically no matter how large their lobes are.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# symbol -> (Z, covalent radius / Angstrom, default colour)
ELEMENTS = {
    "H": (1, 0.31, "#ffffff"), "He": (2, 0.28, "#d9ffff"), "Li": (3, 1.28, "#cc80ff"),
    "B": (5, 0.84, "#ffb5b5"), "C": (6, 0.76, "#404040"), "N": (7, 0.71, "#3050f8"),
    "O": (8, 0.66, "#ff0d0d"), "F": (9, 0.57, "#90e050"), "Si": (14, 1.11, "#f0c8a0"),
    "P": (15, 1.07, "#ff8000"), "S": (16, 1.05, "#e6c619"), "Cl": (17, 1.02, "#1ff01f"),
    "Br": (35, 1.20, "#a62929"), "I": (53, 1.39, "#940094"), "Fe": (26, 1.32, "#e06633"),
    "Co": (27, 1.26, "#f090a0"), "Ni": (28, 1.24, "#50d050"), "Cu": (29, 1.32, "#c88033"),
    "Zn": (30, 1.22, "#7d80b0"), "Au": (79, 1.36, "#ffd123"),
}
Z_TO_SYMBOL = {v[0]: k for k, v in ELEMENTS.items()}


@dataclass
class RenderStyle:
    window_size: tuple = (1600, 1200)
    background: str = "white"
    positive_color: str = "#1f4fd1"     # blue
    negative_color: str = "#d62728"     # red
    orbital_opacity: float = 0.85
    orbital_specular: float = 0.3
    smooth_shading: bool = True
    element_colors: dict = field(default_factory=dict)   # overrides, e.g. {"C": "#333333"}
    atom_radius_scale: float = 0.40     # sphere radius = scale x covalent radius
    bond_radius: float = 0.09           # Angstrom
    bond_tolerance: float = 1.15        # bonded if d < tol x (r_cov,i + r_cov,j)
    bond_color: str | None = None       # None -> half-bonds coloured like their atoms
    sphere_resolution: int = 24
    cylinder_resolution: int = 16
    anti_aliasing: str | None = "ssaa"  # "ssaa", "msaa", "fxaa" or None

    def color(self, symbol):
        return self.element_colors.get(symbol, ELEMENTS.get(symbol, (0, 0.75, "#ff69b4"))[2])


@dataclass
class Camera:
    position: tuple
    focal_point: tuple
    view_up: tuple
    parallel_scale: float
    clipping_range: tuple

    def to_dict(self):
        return {k: list(v) if isinstance(v, (tuple, list, np.ndarray)) else v for k, v in asdict(self).items()}

    @staticmethod
    def from_dict(d):
        return Camera(tuple(d["position"]), tuple(d["focal_point"]), tuple(d["view_up"]),
                      float(d["parallel_scale"]), tuple(d["clipping_range"]))


def frame_camera(points, margin, window_size, distance=100.0):
    """
    Orthographic camera looking down -z of the display frame, framing `points`
    (atoms of every geometry in the comparison set) plus `margin` Angstrom for lobes.
    """
    P = np.asarray(points, dtype=float)
    lo, hi = P.min(axis=0), P.max(axis=0)
    c = (lo + hi) / 2
    aspect = window_size[0] / window_size[1]
    half_h = (hi[1] - lo[1]) / 2 + margin
    half_w = (hi[0] - lo[0]) / 2 + margin
    scale = max(half_h, half_w / aspect)
    depth = (hi[2] - lo[2]) / 2 + margin + 10.0
    return Camera(position=tuple(c + [0, 0, distance]), focal_point=tuple(c), view_up=(0.0, 1.0, 0.0),
                  parallel_scale=float(scale), clipping_range=(distance - depth, distance + depth))


def camera_contains(camera, points, window_size):
    """True if all points fall inside the camera's orthographic view rectangle."""
    P = np.asarray(points) - np.asarray(camera.focal_point)
    aspect = window_size[0] / window_size[1]
    return bool(np.all(np.abs(P[:, 1]) <= camera.parallel_scale) and
                np.all(np.abs(P[:, 0]) <= camera.parallel_scale * aspect))


def symbols_to_numbers(symbols):
    out = []
    for s in symbols:
        s = s.capitalize()
        if s not in ELEMENTS:
            raise KeyError(f"element {s!r} missing from render.ELEMENTS")
        out.append(ELEMENTS[s][0])
    return np.array(out)


def find_bonds(symbols, coords, tolerance):
    r = np.array([ELEMENTS.get(s, (0, 0.75, ""))[1] for s in symbols])
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    lim = tolerance * (r[:, None] + r[None, :])
    i, j = np.nonzero(np.triu((d < lim) & (d > 0.1), k=1))
    return list(zip(i.tolist(), j.tolist()))


def molecule_meshes(symbols, coords, style):
    """[(mesh, colour)] for atoms (spheres) and bonds (cylinders), coordinates in Angstrom."""
    import pyvista as pv

    coords = np.asarray(coords, dtype=float)
    meshes = []
    unit = pv.Sphere(radius=1.0, theta_resolution=style.sphere_resolution, phi_resolution=style.sphere_resolution)
    for sym in sorted(set(symbols)):
        mask = np.array([s == sym for s in symbols])
        pts = pv.PolyData(coords[mask])
        pts["radius"] = np.full(mask.sum(), style.atom_radius_scale * ELEMENTS.get(sym, (0, 0.75, ""))[1])
        meshes.append((pts.glyph(geom=unit, scale="radius", orient=False, factor=1.0), style.color(sym)))
    by_color = {}
    for i, j in find_bonds(symbols, coords, style.bond_tolerance):
        a, b = coords[i], coords[j]
        mid = (a + b) / 2
        halves = [(a, mid, symbols[i]), (mid, b, symbols[j])]
        if style.bond_color:
            halves = [(a, b, None)]
        for p, q, sym in halves:
            v = q - p
            cyl = pv.Cylinder(center=(p + q) / 2, direction=v, radius=style.bond_radius,
                              height=float(np.linalg.norm(v)), resolution=style.cylinder_resolution, capping=False)
            by_color.setdefault(style.bond_color or style.color(sym), []).append(cyl)
    for color, cyls in by_color.items():
        meshes.append((pv.merge(cyls), color))
    return meshes


def lobe_mesh(vertices, faces):
    import pyvista as pv

    f = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
    mesh = pv.PolyData(np.asarray(vertices, dtype=float), f)
    return mesh.compute_normals(cell_normals=False, point_normals=True, split_vertices=False,
                                consistent_normals=True, auto_orient_normals=False)


def render_orbital(png_path, mol_meshes, lobes, camera, style):
    """
    Render one image. `lobes` maps 'positive'/'negative' to (vertices, faces) already
    in the display frame (or None). Returns the camera state read back AFTER rendering,
    so callers/tests can verify nothing re-framed the view.
    """
    import pyvista as pv

    pl = pv.Plotter(off_screen=True, window_size=list(style.window_size), lighting="light_kit")
    try:
        pl.set_background(style.background)
        for mesh, color in mol_meshes:
            pl.add_mesh(mesh, color=color, smooth_shading=True, specular=0.2)
        for name, color in (("positive", style.positive_color), ("negative", style.negative_color)):
            lobe = lobes.get(name)
            if lobe is None:
                continue
            pl.add_mesh(lobe_mesh(*lobe), color=color, opacity=style.orbital_opacity,
                        smooth_shading=style.smooth_shading, specular=style.orbital_specular)
        if style.orbital_opacity < 1:
            pl.enable_depth_peeling()
        if style.anti_aliasing:
            pl.enable_anti_aliasing(style.anti_aliasing)
        pl.camera_position = [camera.position, camera.focal_point, camera.view_up]
        pl.enable_parallel_projection()
        pl.camera.parallel_scale = camera.parallel_scale
        pl.camera.clipping_range = camera.clipping_range
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        pl.screenshot(str(png_path))
        cam = pl.camera
        return {"position": tuple(cam.position), "focal_point": tuple(cam.focal_point),
                "view_up": tuple(cam.up), "parallel_projection": bool(cam.parallel_projection),
                "parallel_scale": float(cam.parallel_scale), "window_size": tuple(pl.window_size)}
    finally:
        pl.close()


def contact_sheet(rows, title, out_path, panel_width_in=3.4):
    """
    rows: [(row_label, [(image_path | None, caption), ...]), ...]
    Captions sit above each panel, never on top of the molecule.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    nrow, ncol = len(rows), max(len(r[1]) for r in rows)
    aspect = None
    for _, panels in rows:
        for img, _ in panels:
            if img is not None and Path(img).exists():
                with Image.open(img) as im:
                    aspect = im.height / im.width
                break
        if aspect:
            break
    aspect = aspect or 0.75
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel_width_in * ncol + 0.6, panel_width_in * aspect * nrow + 0.9 * nrow + 0.5),
                             squeeze=False)
    for r, (row_label, panels) in enumerate(rows):
        for c in range(ncol):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#cccccc")
            if c >= len(panels):
                ax.axis("off")
                continue
            img, caption = panels[c]
            if img is not None and Path(img).exists():
                ax.imshow(Image.open(img))
            else:
                ax.set_facecolor("#f4f4f4")
                ax.text(0.5, 0.5, "not available", ha="center", va="center", color="#888888",
                        transform=ax.transAxes, fontsize=9)
            ax.set_title(caption, fontsize=9)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=11)
    import textwrap

    width_chars = int(fig.get_figwidth() * 11)
    fig.suptitle("\n".join(textwrap.wrap(title, width_chars)), fontsize=10)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
