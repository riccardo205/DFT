#!/usr/bin/env python3
import sys
import re
from pathlib import Path
import pymol2

# ─── 1) USER CONFIG ────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
ROOT_DIR   = REPO_ROOT / "runs" / "orca" / "short-2OS" / "angle_scan" 
OUTPUT_DIR = REPO_ROOT / "visualisation" / "pymol_rendered_cmap" / "orca" / "angle_scan"
ROTATIONS  = [("x", 0), ("y", 0)]  # e.g. [("y", 90)] to rotate camera before saving
PNG_WIDTH  = 800
PNG_DPI    = 500
# ────────────────────────────────────────────────────────────────────────────────

MULLIKEN_BLOCK_RE = re.compile(
    r"MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS\s*\n-+\n(.*?)\n"
    r"Sum of atomic spin populations:\s*(-?\d+\.\d+)",
    re.S,
)
ATOM_LINE_RE = re.compile(r"\s*(\d+)\s+(\S+)\s*:\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)")


def find_orca_jobs(root: Path):
    """
    Recursively find ORCA output files under root. A .out file is only kept if a
    sibling .inp with the same stem exists (this filters out ORCA's auxiliary
    per-atom outputs, e.g. "high_spin_atom79.out", and any flat folders of .out
    files copied back without their .inp/.xyz siblings).
    """
    jobs = []
    for out_file in sorted(root.rglob("*.out")):
        if out_file.with_suffix(".inp").exists():
            jobs.append(out_file)
    return jobs


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


def find_geometry_file(out_file: Path):
    """
    Pick the geometry to render:
      - relaxation jobs: ORCA writes "<stem>.xyz" with the final (or last
        completed, if not converged) optimized Cartesian coordinates.
      - single-point jobs (functional scan, field scan, ...): geometry never
        changes, so fall back to the input "geometry.xyz".
    """
    optimized = out_file.with_suffix(".xyz")
    if optimized.exists():
        return optimized

    geometry = out_file.parent / "geometry.xyz"
    if geometry.exists():
        return geometry

    return None


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
        cmd.set_color("carbonGrey", [0.6, 0.6, 0.6])
        base_view = cmd.get_view()

        for out_file in jobs:
            relative = out_file.relative_to(root).with_suffix("")
            name = str(relative).replace("/", "_")

            geometry_file = find_geometry_file(out_file)
            if geometry_file is None:
                print(f"[!] No geometry (.xyz) found for {out_file}, skipping.")
                continue

            spins = parse_final_spin_populations(
                out_file.read_text(errors="ignore")
            )
            if not spins:
                print(f"[!] No Mulliken spin populations found in {out_file}")

            cmd.set_view(base_view)
            cmd.load(str(geometry_file), name)
            cmd.show("sticks", name)
            cmd.show("spheres", name)
            cmd.set("sphere_scale", 0.20)
            cmd.color("carbonGrey", f"model {name} and elem C")
            cmd.color("white", f"model {name} and elem H")
            cmd.orient(name)
            if ROTATIONS:
                apply_rotations(cmd, ROTATIONS)

            color_by_spin_colormap(cmd, name, spins)

            out_png = out / relative.with_suffix(".png")
            save_png(cmd, name, out_png, width=PNG_WIDTH, dpi=PNG_DPI)
            cmd.delete(name)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        ROOT_DIR, OUTPUT_DIR = map(Path, sys.argv[1:])
    print(f"Rendering {ROOT_DIR} -> {OUTPUT_DIR}")
    main(ROOT_DIR, OUTPUT_DIR)
