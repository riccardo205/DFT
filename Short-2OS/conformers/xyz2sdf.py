#!/usr/bin/env python3
import argparse
import os
from openbabel import pybel

def xyz_dir_to_sdf(xyz_dir: str, sdf_path: str):
    if not os.path.isdir(xyz_dir):
        raise FileNotFoundError(f"Input folder '{xyz_dir}' not found")

    # Collect all xyz files (sorted for consistency)
    xyz_files = sorted(
        f for f in os.listdir(xyz_dir)
        if f.lower().endswith('.xyz')
    )
    if not xyz_files:
        raise RuntimeError(f"No .xyz files found in '{xyz_dir}'")

    # Prepare SDF writer
    writer = pybel.Outputfile("sdf", sdf_path, overwrite=True)

    for fname in xyz_files:
        fullpath = os.path.join(xyz_dir, fname)
        # readfile yields a generator; most .xyz contain a single molecule
        for mol in pybel.readfile("xyz", fullpath):
            # set the title to the basename (without extension)
            mol.title = os.path.splitext(fname)[0]
            writer.write(mol)

    writer.close()
    print(f"Wrote {len(xyz_files)} molecules to '{sdf_path}'")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Merge all .xyz in a folder into one multi-molecule .sdf"
    )
    p.add_argument("xyz_folder", help="Path to folder containing .xyz files")
    p.add_argument("output_sdf", help="Path to output .sdf file")
    args = p.parse_args()

    xyz_dir_to_sdf(args.xyz_folder, args.output_sdf)
