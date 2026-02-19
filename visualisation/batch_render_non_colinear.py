#!/usr/bin/env python3
import sys
import csv
from pathlib import Path
import pymol2

# ─── 1) USER CONFIG ────────────────────────────────────────────────────────────
initial_label = "dimers_run2/non_colinear_fast2"
label_out     = "dimers_run2/non_colinear_fast2"
ROOT_DIR      = Path(f"Short-2OS/runs/{initial_label}")
OUTPUT_DIR    = Path(f"visualisation/pymol_rendered_cmap/{label_out}")
CSV_PATH      = Path(f"visualisation/{initial_label}/all_spins.csv")
ROTATIONS     = [("x", 0),("y", 0)]  # e.g. [("y", 90)] to rotate camera before saving
PNG_WIDTH     = 800
PNG_DPI       = 500
# ────────────────────────────────────────────────────────────────────────────────

def load_spin_map(csv_path):
    """Return a dict: molecule_name -> {Sx,Sy,Sz} spin strings"""
    spin_map = {}
    if csv_path is not None:
        with open(csv_path, newline='') as csvfile:
            for row in csv.DictReader(csvfile):
                name = row["Folder"].strip()
                spin_map[name] = {
                    "Sx": row.get("Spin_final_all_Sx", "").strip(),
                    "Sy": row.get("Spin_final_all_Sy", "").strip(),
                    "Sz": row.get("Spin_final_all_Sz", "").strip(),
                }
    return spin_map

def color_by_spin_colormap(cmd, model_name, spin_str, *, pos_color, neg_color):
    """
    spin_str like "6:-0.210; 22:0.181; …"
    Maps each atom index to a color on a white→red (pos) / white→blue (neg) gradient.
    """
    # parse into list of (idx, intensity)
    items = []
    for token in spin_str.split(';'):
        token = token.strip()
        if not token:
            continue
        idx_s, val_s = token.split(':')
        items.append((int(idx_s), float(val_s)))

    if not items:
        return

    # find maximum absolute intensity for scaling
    max_abs = max(abs(val) for _, val in items)
    if max_abs == 0:
        max_abs = 1.0
        
    carbonGrey = (0.6, 0.6, 0.6)
    pos_r, pos_g, pos_b = pos_color
    neg_r, neg_g, neg_b = neg_color
    for atom_idx, intensity in items:
        t = abs(intensity) / max_abs
        if intensity >= 0:
            # grey → positive color
            r = carbonGrey[0] + t * (pos_r - carbonGrey[0])
            g = carbonGrey[1] + t * (pos_g - carbonGrey[1])
            b = carbonGrey[2] + t * (pos_b - carbonGrey[2])
        else:
            # grey → negative color
            r = carbonGrey[0] + t * (neg_r - carbonGrey[0])
            g = carbonGrey[1] + t * (neg_g - carbonGrey[1])
            b = carbonGrey[2] + t * (neg_b - carbonGrey[2])

        color_name = f"spin_{model_name}_{atom_idx}"
        cmd.set_color(color_name, [r, g, b])
        sel = f"model {model_name} and index {atom_idx}"
        cmd.color(color_name, sel)


def apply_rotations(cmd, rotations):
    """Rotate the camera by a list of (axis, angle_deg) pairs."""
    for axis, angle in rotations:
        cmd.turn(axis, angle)


def save_png(cmd, obj_name, out_dir, width, dpi, suffix=""):
    out_png = (out_dir / f"{obj_name}{suffix}").with_suffix(".png")
    cmd.png(str(out_png), width=width, dpi=dpi, ray=1)
    print(f"✅ {obj_name} → {out_png.name}")

def main(root: Path, out: Path, csv_path: Path):
    out.mkdir(parents=True, exist_ok=True)
    spin_map = load_spin_map(csv_path)

    with pymol2.PyMOL() as pymol:
        cmd = pymol.cmd
        cmd.bg_color("white")
        # base colors
        cmd.set_color('carbonGrey', [0.6, 0.6, 0.6])
        base_view = cmd.get_view()

        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue

            xyzs = list(sub.glob("*.xyz"))
            if not xyzs:
                print(f"[!] No .xyz in {sub.name}, skipping.")
                continue

            xyz_path = xyzs[0]
            name     = sub.name

            cmd.set_view(base_view)

            cmd.load(str(xyz_path), name)
            cmd.show('sticks',  name)
            cmd.show('spheres', name)
            cmd.set('sphere_scale', 0.20)
            cmd.color('carbonGrey',      f'model {name} and elem C')
            cmd.orient(name)
            if ROTATIONS:
                apply_rotations(cmd, ROTATIONS)
            
            spins = spin_map.get(name, {})

            # Render Sx (green/magenta)
            cmd.color('carbonGrey', f'model {name} and elem C')
            spin_str = spins.get("Sx", "")
            if spin_str:
                color_by_spin_colormap(
                    cmd,
                    name,
                    spin_str,
                    pos_color=(0.0, 0.8, 0.2),
                    neg_color=(0.6, 0.2, 0.8),
                )
            cmd.color('white', f'model {name} and elem H')
            save_png(cmd, name, out, width=PNG_WIDTH, dpi=PNG_DPI, suffix="_Sx")

            # Render Sy (orange/cyan)
            cmd.color('carbonGrey', f'model {name} and elem C')
            spin_str = spins.get("Sy", "")
            if spin_str:
                color_by_spin_colormap(
                    cmd,
                    name,
                    spin_str,
                    pos_color=(1.0, 0.6, 0.0),
                    neg_color=(0.0, 0.7, 0.8),
                )
            cmd.color('white', f'model {name} and elem H')
            save_png(cmd, name, out, width=PNG_WIDTH, dpi=PNG_DPI, suffix="_Sy")

            # Render Sz (red/blue, same as before)
            cmd.color('carbonGrey', f'model {name} and elem C')
            spin_str = spins.get("Sz", "")
            if spin_str:
                color_by_spin_colormap(
                    cmd,
                    name,
                    spin_str,
                    pos_color=(1.0, 0.0, 0.0),
                    neg_color=(0.0, 0.0, 1.0),
                )
            cmd.color('white', f'model {name} and elem H')
            save_png(cmd, name, out, width=PNG_WIDTH, dpi=PNG_DPI, suffix="_Sz")

            cmd.delete(name)

if __name__ == "__main__":
    if len(sys.argv) == 4:
        ROOT_DIR, OUTPUT_DIR, CSV_PATH = map(Path, sys.argv[1:])
    print(f"Rendering {ROOT_DIR} → {OUTPUT_DIR} using {CSV_PATH}")
    main(ROOT_DIR, OUTPUT_DIR, CSV_PATH)
