#!/usr/bin/env python3
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
    # Set input/output paths here
    xyz_folder = "Gold_electrodes/main_conf/junction_SM_test"  # Folder containing .xyz files
    output_sdf = "sdf_files/junction_SM_test.sdf"

    xyz_dir_to_sdf(xyz_folder, output_sdf)
