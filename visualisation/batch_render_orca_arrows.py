#!/usr/bin/env python3
"""
Copy of batch_render_orca.py with spin arrows (as in batch_render_arrows.py): every atom with
|Mulliken spin| > ARROW_THRESHOLD gets an arrow pointing up on screen (red, spin > 0) or down (blue,
spin < 0), length proportional to |spin| / max|spin|. Everything else (spin colours, molecule planes,
bond colours, A-D labels, legend) is the same as batch_render_orca.py.
"""
import io
import json
import re
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pymol2
from ase.io import read as ase_read
from ase.neighborlist import NeighborList, natural_cutoffs
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pymol.cgo import ALPHA, BEGIN, COLOR, CONE, CYLINDER, END, NORMAL, TRIANGLES, VERTEX

# ─── 1) USER CONFIG ────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
ROOT_DIR   = REPO_ROOT / "runs" / "orca" / "short-2OS" / "dimers_run2" / "run2_broken_symmetry" / "relaxed" /"conf_11"
OUTPUT_DIR = REPO_ROOT / "visualisation" / "pymol_rendered_cmap" / "orca" / "dimers_run2" / "run2_broken_symmetry" / "conf_11_arrows"
ROTATIONS  = [("x", 0), ("y", 0)]  # e.g. [("y", 90)] to rotate camera before saving
PNG_WIDTH  = 800
PNG_DPI    = 500
# molecule identification (spheres keep the spin colour; planes, bonds and labels show the molecule)
MOLECULE_COLORS   = [(0.11, 0.62, 0.42), (0.48, 0.31, 0.80), (0.85, 0.55, 0.00), (0.35, 0.35, 0.35)]  # green, purple, ...
SHOW_RING_PLANES  = True    # translucent polygon on every ring, coloured by molecule
PLANE_ALPHA       = 0.15    # opacity of the ring planes of the radical units (ring systems with a radical carbon)
LINKER_ALPHA      = 0.6    # opacity of the linker planes (ring systems without a radical carbon); 0 = no plane
LINKER_LIGHTEN    = 0.6     # 0..1, mix with white for ring systems without a radical carbon (anthryl linkers)
BONDS_BY_MOLECULE = True    # sticks in a dark shade of the molecule colour, spheres coloured by spin
LABEL_RADICALS    = False   # label A-D at the radical carbons (needs radical_fragments.json near the job)
ADD_LEGEND        = False   # legend strip appended below every PNG
STICK_RADIUS      = 0.12    # thinner sticks so the spin-coloured spheres dominate
STICK_TRANSPARENCY        = 0.3      # 0 = opaque, 1 = invisible; all bonds
LINKER_STICK_TRANSPARENCY = 0.5    # e.g. 0.8: separate value for bonds inside linker ring systems; None = same as above
SPHERE_SCALE      = 0.15
LABEL_OFFSET_PX   = 40      # px, distance of the A-D labels from their radical carbon (away from the dimer centre)
LABEL_FONTSIZE    = 16
# spin arrows
SHOW_ARROWS       = True
ARROW_THRESHOLD   = 0.15    # draw an arrow where |Mulliken spin| > threshold (radical C9 ~0.5)
ARROW_LENGTH      = 1.8     # Å, length of the arrow for the largest |spin|
ARROW_RADIUS      = 0.12    # Å, shaft radius (head radius = 2.1 x)
ARROW_UP_COLOR    = (0.85, 0.0, 0.0)
ARROW_DOWN_COLOR  = (0.0, 0.0, 0.85)
HIDE_HYDROGENS    = True   # True: do not draw H atoms (geometry, spins and fragments are unchanged)
# ────────────────────────────────────────────────────────────────────────────────

MULLIKEN_BLOCK_RE = re.compile(
    r"MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS\s*\n-+\n(.*?)\n"
    r"Sum of atomic spin populations:\s*(-?\d+\.\d+)",
    re.S,
)
ATOM_LINE_RE = re.compile(r"\s*(\d+)\s+(\S+)\s*:\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)")
COORD_BLOCK_RE = re.compile(
    r"CARTESIAN COORDINATES \(ANGSTROEM\)\s*\n-+\n((?:\s*[A-Z][a-z]?\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s*\n)+)"
)
ORCA_HEADER = "O   R   C   A"
AUX_OUT_RE = re.compile(r"(_atom\d+|^slurm)\.out$")   # ORCA per-atom outputs, slurm logs


def is_orca_main_output(out_file: Path):
    """True for an ORCA main output: either it has a sibling .inp with the same stem, or (for
    .out files copied on their own, e.g. selected_out_files/<dimer>/uuuu.out) it starts with the
    ORCA banner. ORCA's per-atom auxiliary outputs ("high_spin_atom79.out") and slurm logs are
    always skipped."""
    if AUX_OUT_RE.search(out_file.name):
        return False
    if out_file.with_suffix(".inp").exists():
        return True
    with out_file.open(errors="ignore") as fh:
        return ORCA_HEADER in fh.read(4000)


def find_orca_jobs(root: Path):
    """
    Recursively find ORCA main output files under root, at any depth: root can be one
    calculation folder, a folder of calculation folders (e.g. <dimer>/<config>/), or deeper
    trees (<run>/<stage>/<dimer>/<config>/). Natural order: conf_2 before conf_10.
    """
    natural = lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(p))]
    return [p for p in sorted(root.rglob("*.out"), key=natural) if is_orca_main_output(p)]


def parse_final_spin_populations(out_text: str):
    """
    Return [(atom_index, spin), ...] for the LAST "MULLIKEN ATOMIC CHARGES AND
    SPIN POPULATIONS" block in the file, i.e. the final geometry / final SCF
    cycle (for a relaxation this is the last completed optimization step,
    converged or not).
    """
    blocks = list(MULLIKEN_BLOCK_RE.finditer(out_text))
    if not blocks:
        return []

    block_text, _ = blocks[-1].groups()
    spins = []
    for line in block_text.splitlines():
        m = ATOM_LINE_RE.match(line)
        if not m:
            continue
        atom_idx = int(m.group(1))
        spin = float(m.group(4))
        spins.append((atom_idx, spin))
    return spins


def find_geometry(out_file: Path, out_text: str):
    """
    Geometry to render, as (description, xyz text):
      - relaxation jobs: ORCA writes "<stem>.xyz" with the final (or last
        completed, if not converged) optimized Cartesian coordinates.
      - single-point jobs (functional scan, field scan, ...): geometry never
        changes, so fall back to the input "geometry.xyz".
      - .out copied without its folder (no .xyz next to it): the LAST
        "CARTESIAN COORDINATES (ANGSTROEM)" block of the output itself, i.e. the
        final geometry of a relaxation or the geometry of a single point.
    Returns (None, None) if nothing is found.
    """
    optimized = out_file.with_suffix(".xyz")
    if optimized.exists():
        return optimized.name, optimized.read_text()

    # geometry.xyz belongs to this job only in a real calculation folder (sibling .inp);
    # in a flat folder of copied outputs it could belong to another job
    geometry = out_file.parent / "geometry.xyz"
    if geometry.exists() and out_file.with_suffix(".inp").exists():
        return geometry.name, geometry.read_text()

    blocks = COORD_BLOCK_RE.findall(out_text)
    if blocks:
        lines = [ln.split() for ln in blocks[-1].strip().splitlines()]
        xyz = f"{len(lines)}\n{out_file.name}\n" + "\n".join(" ".join(ln) for ln in lines) + "\n"
        return "coordinates in .out", xyz

    return None, None


def color_by_spin_colormap(cmd, model_name, spins):
    """
    spins: [(atom_index, sz), ...] with ORCA's 0-based atom indexing.
    Maps each atom to a color on a grey->red (positive) / grey->blue (negative)
    gradient, scaled by the largest |sz| in this structure. PyMOL's `index`
    selector is 1-based, hence the +1 offset below.
    """
    if not spins:
        return

    max_abs = max(abs(val) for _, val in spins)
    if max_abs == 0:
        max_abs = 1.0

    carbon_grey = (0.6, 0.6, 0.6)
    for atom_idx, intensity in spins:
        t = abs(intensity) / max_abs
        if intensity >= 0:
            r = carbon_grey[0] + t * (1.0 - carbon_grey[0])
            g = carbon_grey[1] * (1.0 - t)
            b = carbon_grey[2] * (1.0 - t)
        else:
            r = carbon_grey[0] * (1.0 - t)
            g = carbon_grey[1] * (1.0 - t)
            b = carbon_grey[2] + t * (1.0 - carbon_grey[2])

        color_name = f"spin_{model_name}_{atom_idx}"
        cmd.set_color(color_name, [r, g, b])
        sel = f"model {model_name} and index {atom_idx + 1}"
        cmd.color(color_name, sel)


def shade(rgb, white=0.0, black=0.0):
    rgb = np.array(rgb, float)
    rgb = rgb + white * (1 - rgb)
    return tuple(rgb * (1 - black))


def molecular_structure(xyz_text):
    """Molecules (bonded components, sorted by first atom), carbon/heavy ring systems (fused rings sharing a
    bond) and positions, from the geometry alone. Indices are 0-based (ORCA order)."""
    atoms = ase_read(io.StringIO(xyz_text), format="xyz")
    nl = NeighborList(natural_cutoffs(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)
    g = nx.Graph()
    g.add_nodes_from(range(len(atoms)))
    for i in range(len(atoms)):
        g.add_edges_from((i, int(j)) for j in nl.get_neighbors(i)[0] if j > i)
    molecules = sorted((sorted(c) for c in nx.connected_components(g)), key=min)
    heavy = g.subgraph([i for i in g if atoms[i].symbol != "H"])
    rings = [sorted(r) for r in nx.minimum_cycle_basis(heavy)]
    fused = nx.Graph()
    fused.add_nodes_from(range(len(rings)))
    fused.add_edges_from((a, b) for a in range(len(rings)) for b in range(a + 1, len(rings))
                         if len(set(rings[a]) & set(rings[b])) >= 2)
    systems = [[rings[k] for k in comp] for comp in nx.connected_components(fused)]
    mol_of = {i: m for m, mol in enumerate(molecules) for i in mol}
    return {"pos": atoms.positions, "symbols": atoms.get_chemical_symbols(), "molecules": molecules,
            "systems": systems, "mol_of": mol_of}


def find_radical_fragments(out_file: Path, n_atoms):
    """radical_fragments.json of the dimer (written by prepare_dimers_broken_symmetry.py /
    prepare_run4_broken_symmetry.py), searched in the job folder and up to two levels above."""
    for d in (out_file.parent, out_file.parent.parent, out_file.parent.parent.parent):
        f = d / "radical_fragments.json"
        if f.exists():
            frags = json.loads(f.read_text()).get("fragments", {})
            if frags and max(max(v) for v in frags.values()) < n_atoms:
                return frags
    return None


def ordered_ring(pos, ring):
    """Ring atoms in cyclic order (angle around the centroid in the ring's best plane)."""
    p = pos[ring]
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c)
    ang = np.arctan2((p - c) @ vt[1], (p - c) @ vt[0])
    return [ring[k] for k in np.argsort(ang)], c, vt[2]


def ring_planes_cgo(struct, radical_carbons):
    """CGO triangles filling every ring; colour = molecule, lighter for ring systems without a radical carbon."""
    pos, cgo = struct["pos"], []
    for system in struct["systems"]:
        atoms_sys = set().union(*system)
        mol = struct["mol_of"][next(iter(atoms_sys))]
        base = MOLECULE_COLORS[mol % len(MOLECULE_COLORS)]
        radical_unit = bool(atoms_sys & radical_carbons) or not radical_carbons
        col = base if radical_unit else shade(base, white=LINKER_LIGHTEN)
        alpha = PLANE_ALPHA if radical_unit else LINKER_ALPHA
        if alpha <= 0:
            continue
        for ring in system:
            order, c, n = ordered_ring(pos, ring)
            cgo += [ALPHA, alpha, COLOR, *col, BEGIN, TRIANGLES, NORMAL, *n]
            for k in range(len(order)):
                cgo += [VERTEX, *c, VERTEX, *pos[order[k]], VERTEX, *pos[order[(k + 1) % len(order)]]]
            cgo += [END]
    return cgo


def index_selection(model, atoms):
    """PyMOL selection for 0-based atom indices, compressed into ranges."""
    atoms = sorted(atoms)
    parts, start = [], atoms[0]
    for a, b in zip(atoms, atoms[1:] + [None]):
        if b != a + 1:
            parts.append(f"{start + 1}-{a + 1}" if a > start else f"{a + 1}")
            start = b
    return f"model {model} and index " + "+".join(parts)


def decorate_molecules(cmd, name, struct, spins, frags):
    """Planes, bond colours and radical labels; returns the radical labels per molecule for the legend."""
    spin = dict(spins)
    radical_carbons, labels = set(), {}
    if frags:
        for lab, half in sorted(frags.items()):
            carbons = [i for i in half if struct["symbols"][i] == "C"]
            c9 = max(carbons, key=lambda i: abs(spin.get(i, 0.0))) if spin else None
            if c9 is not None and abs(spin.get(c9, 0.0)) > 0.05:
                radical_carbons.add(c9)
            labels[lab] = (c9, struct["mol_of"][half[0]])
    extra = []
    if SHOW_RING_PLANES and struct["systems"]:
        cmd.load_cgo(ring_planes_cgo(struct, radical_carbons), f"{name}_planes")
        extra.append(f"{name}_planes")
    for m, mol in enumerate(struct["molecules"]):
        base = MOLECULE_COLORS[m % len(MOLECULE_COLORS)]
        cmd.set_color(f"mol{m}_{name}", list(base))
        cmd.set_color(f"mol{m}_dark_{name}", list(shade(base, black=0.15)))
        if BONDS_BY_MOLECULE:
            cmd.set("stick_color", f"mol{m}_dark_{name}", index_selection(name, mol))
    if STICK_TRANSPARENCY > 0:
        cmd.set("stick_transparency", STICK_TRANSPARENCY, name)
    if LINKER_STICK_TRANSPARENCY is not None and radical_carbons:
        linker = sorted(set().union(*[set().union(*sys_) for sys_ in struct["systems"]
                                      if not set().union(*sys_) & radical_carbons]))
        if linker:      # bond-level setting: bonds with both atoms in a linker ring system
            cmd.set_bond("stick_transparency", LINKER_STICK_TRANSPARENCY, index_selection(name, linker))
    label_points = []                       # drawn on the PNG afterwards (PyMOL ray-traced labels get occluded)
    if LABEL_RADICALS:
        for lab, (c9, m) in labels.items():
            if c9 is not None:
                label_points.append((lab, struct["pos"][c9], shade(MOLECULE_COLORS[m % len(MOLECULE_COLORS)], black=0.15)))
    mol_labels = {m: [lab for lab, (_, mm) in sorted(labels.items()) if mm == m] for m in range(len(struct["molecules"]))}
    return extra, mol_labels, bool(radical_carbons), label_points


def project_to_pixels(cmd, points, width, height):
    """Pixel coordinates (x right, y down) of model-space points in the current PyMOL view."""
    view = np.array(cmd.get_view())
    rot = view[:9].reshape(3, 3)                                   # model -> camera: v_cam = v @ rot
    cam = (np.atleast_2d(points) - view[12:15]) @ rot + view[9:12]
    t = np.tan(np.radians(float(cmd.get("field_of_view"))) / 2)
    depth = -view[11] if cmd.get("orthoscopic") in ("on", "1") else -cam[:, 2]
    scale = (height / 2) / (t * depth)
    return np.column_stack([width / 2 + cam[:, 0] * scale, height / 2 - cam[:, 1] * scale])


def annotate_png(png: Path, label_px, centre_px, mol_labels, has_radicals, legend):
    """Draw the A-D labels (offset away from the dimer centre in the image) and the legend strip."""
    img = plt.imread(png)
    h, w = img.shape[:2]
    extra = 0.55 if legend else 0.0
    fig = plt.figure(figsize=(w / 100, h / 100 + extra), dpi=100)
    ax = fig.add_axes([0, extra / (h / 100 + extra), 1, h / 100 / (h / 100 + extra)])
    ax.imshow(img)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    for lab, (x, y), col in label_px:
        d = np.array([x, y]) - centre_px
        d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1 else np.array([0.0, -1.0])
        tx, ty = np.clip([x + LABEL_OFFSET_PX * d[0], y + LABEL_OFFSET_PX * d[1]], [12, 12], [w - 12, h - 12])
        ax.annotate(lab, (x, y), (tx, ty), ha="center", va="center", fontsize=LABEL_FONTSIZE, fontweight="bold",
                    color=col, path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                    arrowprops=dict(arrowstyle="-", color=col, lw=1.0, shrinkA=8, shrinkB=4))
    if legend:
        handles = [Patch(facecolor=MOLECULE_COLORS[m % len(MOLECULE_COLORS)], alpha=0.8,
                         label=f"molecule {m + 1}" + (f" ({', '.join(l)})" if l else "")) for m, l in mol_labels.items()]
        if has_radicals:
            handles.append(Patch(facecolor="0.85", label="pale: linker (no radical C)"))
        handles += [Line2D([0], [0], marker="o", color="none", markerfacecolor=(1, 0, 0), markersize=7, label="spin > 0"),
                    Line2D([0], [0], marker="o", color="none", markerfacecolor=(0, 0, 1), markersize=7, label="spin < 0")]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, frameon=False,
                   handlelength=1.2, columnspacing=1.0)
    fig.savefig(png, dpi=100)
    plt.close(fig)


def draw_spin_arrows(cmd, name, struct, spins):
    """CGO arrows (cylinder + cone) on every atom with |spin| > ARROW_THRESHOLD, along the screen-up direction
    of the current view (up = spin > 0, down = spin < 0), length ~ |spin|. Returns the CGO object names."""
    if not spins:
        return []
    view = cmd.get_view()
    up = np.array([view[1], view[4], view[7]])           # camera y axis in model coordinates
    up /= np.linalg.norm(up)
    smax = max(abs(v) for _, v in spins) or 1.0
    objs = []
    for idx, s in spins:
        if abs(s) <= ARROW_THRESHOLD or (HIDE_HYDROGENS and struct["symbols"][idx] == "H"):
            continue
        p0 = struct["pos"][idx]
        vec = np.sign(s) * up * ARROW_LENGTH * abs(s) / smax
        head = p0 + 0.65 * vec
        tip = p0 + vec
        col = ARROW_UP_COLOR if s > 0 else ARROW_DOWN_COLOR
        obj = [CYLINDER, *p0, *head, ARROW_RADIUS, *col, *col,
               CONE, *head, *tip, 2.1 * ARROW_RADIUS, 0.0, *col, *col, 1.0, 1.0]
        oname = f"{name}_arrow_{idx}"
        cmd.load_cgo(obj, oname)
        objs.append(oname)
    return objs


def apply_rotations(cmd, rotations):
    for axis, angle in rotations:
        cmd.turn(axis, angle)


def save_png(cmd, obj_name, out_png: Path, width, dpi):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd.png(str(out_png), width=width, dpi=dpi, ray=1)
    print(f"[ok] {obj_name} -> {out_png}")


def main(root: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)

    jobs = find_orca_jobs(root)
    if not jobs:
        print(f"[!] No ORCA .out/.inp pairs found under {root}")
        return

    with pymol2.PyMOL() as pymol:
        cmd = pymol.cmd
        cmd.bg_color("white")
        cmd.set("two_sided_lighting", 1)      # ring planes are lit from both sides (no dark back faces)
        cmd.set_color("carbonGrey", [0.6, 0.6, 0.6])
        base_view = cmd.get_view()

        for out_file in jobs:
            relative = out_file.relative_to(root).with_suffix("")
            name = str(relative).replace("/", "_")

            out_text = out_file.read_text(errors="ignore")
            geometry_source, xyz_text = find_geometry(out_file, out_text)
            if xyz_text is None:
                print(f"[!] No geometry (.xyz or coordinates in the .out) for {out_file}, skipping.")
                continue

            spins = parse_final_spin_populations(out_text)
            if not spins:
                print(f"[!] No Mulliken spin populations found in {out_file}")

            cmd.set_view(base_view)
            with tempfile.NamedTemporaryFile("w", suffix=".xyz", delete=False) as tmp:
                tmp.write(xyz_text)
            cmd.load(tmp.name, name, format="xyz")
            Path(tmp.name).unlink()
            cmd.show("sticks", name)
            cmd.show("spheres", name)
            cmd.set("sphere_scale", SPHERE_SCALE)
            cmd.set("stick_radius", STICK_RADIUS)
            cmd.color("carbonGrey", f"model {name} and elem C")
            cmd.color("white", f"model {name} and elem H")
            if HIDE_HYDROGENS:
                cmd.hide("everything", f"model {name} and elem H")
            cmd.orient(name)
            if ROTATIONS:
                apply_rotations(cmd, ROTATIONS)

            color_by_spin_colormap(cmd, name, spins)
            struct = molecular_structure(xyz_text)
            frags = find_radical_fragments(out_file, len(struct["pos"]))
            view = cmd.get_view()                              # extra CGO objects must not change the zoom
            extra, mol_labels, has_radicals, label_points = decorate_molecules(cmd, name, struct, spins, frags)
            if SHOW_ARROWS:
                extra += draw_spin_arrows(cmd, name, struct, spins)
            cmd.set_view(view)

            out_png = out / relative.with_suffix(".png")       # same folder structure as under root
            save_png(cmd, f"{name} (geometry: {geometry_source})", out_png, width=PNG_WIDTH, dpi=PNG_DPI)
            if ADD_LEGEND or label_points:
                h, w = plt.imread(out_png).shape[:2]
                pts = [p for _, p, _ in label_points] + [struct["pos"].mean(0)]
                px = project_to_pixels(cmd, np.array(pts), w, h)
                label_px = [(lab, px[k], col) for k, (lab, _, col) in enumerate(label_points)]
                annotate_png(out_png, label_px, px[-1], mol_labels, has_radicals, ADD_LEGEND)
            for obj in [name] + extra:
                cmd.delete(obj)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        ROOT_DIR, OUTPUT_DIR = map(Path, sys.argv[1:])
    print(f"Rendering {ROOT_DIR} -> {OUTPUT_DIR}")
    main(ROOT_DIR, OUTPUT_DIR)
