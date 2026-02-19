#!/usr/bin/env python3
import sys
from pathlib import Path
import shutil

def main(fdf_folder: Path, output_dir: Path, *target_files):
    """
    Create folders for each .fdf file (without extension) in output_dir,
    and copy target files into each folder.
    """
    fdf_folder = Path(fdf_folder)
    output_dir = Path(output_dir)
    target_files = [Path(f) for f in target_files]
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all .fdf files
    fdf_files = sorted(fdf_folder.glob("*.fdf"))
    
    if not fdf_files:
        print(f"[!] No .fdf files found in {fdf_folder}")
        return
    
    # Check all target files exist
    for target_file in target_files:
        if not target_file.exists():
            print(f"[!] Target file not found: {target_file}")
            return
    
    for fdf_path in fdf_files:
        folder_name = fdf_path.stem  # Get filename without extension
        new_folder = output_dir / folder_name
        
        # Create folder
        new_folder.mkdir(parents=True, exist_ok=True)
        
        # Copy all target files
        for target_file in target_files:
            dest_file = new_folder / target_file.name
            shutil.copy2(str(target_file), str(dest_file))
        
        files_str = ", ".join(f.name for f in target_files)
        print(f"✅ Created {folder_name} and copied {files_str}")

if __name__ == "__main__":
    fdf_folder = "Short-2OS/molecule_creation/fdf_files/dimers_run2/non_colinear"
    output_dir = "Short-2OS/runs/dimers_run2/non_colinear_fast2"
    target_files = [
        "Short-2OS/runs/dimers_run2/slow_lowT/conf_11_antiferro2/dimers-short-2OS.XV",
        "Short-2OS/runs/dimers_run2/slow_lowT/conf_11_antiferro2/dimers-short-2OS.xyz"
    ]
    
    main(fdf_folder, output_dir, *target_files)
