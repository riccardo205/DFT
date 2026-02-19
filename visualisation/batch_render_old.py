#!/usr/bin/env python3
import sys
import csv
from pathlib import Path
import pymol2

# ─── 1) USER CONFIG ────────────────────────────────────────────────────────────
# label of the run folder
initial_label = "main_conf/fast"
# label of the output render folder
label_out = "main_conf/fdf_files"
# Change this to the folder containing your per-molecule subfolders:
ROOT_DIR = Path(f"Short-2OS/conformers/fdf_files/{initial_label}")
# The script will create OUTPUT_DIR if needed:
OUTPUT_DIR = Path(f"visualisation/pymol_rendered/{label_out}")
# CSV file with the Spin_final_list column
CSV_PATH = Path(f"visualisation/{initial_label}/summary_results.csv")
CSV_PATH = None
# ────────────────────────────────────────────────────────────────────────────────

def load_spin_map(csv_path):
    """Return a dict: folder_name -> spin_final_list string"""
    spin_map = {}
    if csv_path is not None:
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row.get("Folder", "").strip()
                val  = row.get("Spin_final_list", "").strip()
                spin_map[name] = val
    return spin_map

def color_by_spin(cmd, model_name, spin_str):
    """
    spin_str like "6:-0.210; 22:0.181; …"
    Colors each atom index according to intensity.
    """
    for item in spin_str.split(';'):
        item = item.strip()
        if not item:
            continue
        idx_str, inten_str = item.split(':')
        idx       = idx_str.strip()
        intensity = float(inten_str)
        # choose color
        if   intensity >  0.3:  color = 'red'
        elif intensity >  0.0:  color = 'lightRed'
        elif intensity >= -0.3: color = 'lightBlue'
        else:                   color = 'blue'
        # apply it
        sel = f'model {model_name} and index {idx}'
        cmd.color(color, sel)

def main(root: Path, out: Path, csv_path: Path):
    # ensure output dir exists
    out.mkdir(parents=True, exist_ok=True)
    # load the CSV lookup
    spin_map = load_spin_map(csv_path)

    with pymol2.PyMOL() as pymol:
        cmd = pymol.cmd
        cmd.bg_color("white")
        # define a uniform grey for all carbons by default
        cmd.set_color('carbonGrey', [0.3, 0.3, 0.3])
        # define your new light colors
        cmd.set_color('lightRed',  [1.0, 0.6, 0.6])
        cmd.set_color('lightBlue', [0.6, 0.6, 1.0])

        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue

            xyz_files = list(sub.glob("*.xyz"))
            if not xyz_files:
                print(f"[!] No .xyz in {sub.name}, skipping.")
                continue

            xyz_path = xyz_files[0]
            name     = sub.name

            # load & base styling
            cmd.load(str(xyz_path), name)
            cmd.show('sticks', name)
            cmd.show('spheres', name)
            cmd.set('sphere_scale', 0.20)
            cmd.color('carbonGrey', f'model {name} and elem C')
            cmd.color('white',      f'model {name} and elem H')

            # if we have spin data, recolor specific atoms:
            spin_str = spin_map.get(name, "")
            if spin_str:
                color_by_spin(cmd, name, spin_str)

            # render & cleanup
            out_png = out / f"{name}.png"
            cmd.png(str(out_png), width=800, dpi=300, ray=1)
            cmd.delete(name)

            print(f"✅ {name} → {out_png.name}")

if __name__ == "__main__":
    # allow overrides: python script.py ROOT_DIR OUTPUT_DIR CSV_PATH
    if len(sys.argv) == 4:
        ROOT_DIR, OUTPUT_DIR, CSV_PATH = map(Path, sys.argv[1:])
    print(f"Rendering {ROOT_DIR} → {OUTPUT_DIR} using {CSV_PATH}")
    main(ROOT_DIR, OUTPUT_DIR, CSV_PATH)