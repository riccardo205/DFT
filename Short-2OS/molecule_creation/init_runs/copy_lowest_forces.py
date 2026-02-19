#!/usr/bin/env python3
"""
Script to copy .xyz and .xv files from lowest force configurations to other configurations.

This script reads a summary_results.csv file, groups results by config number,
finds the configuration with the lowest F_final for each group, and copies the
corresponding .xyz and .xv files to all other configurations in the same group.
"""

import sys
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import shutil


def read_summary_csv(csv_path: Path) -> List[Dict]:
    """Read the summary_results.csv file and return list of rows."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_folder_name(folder_name: str) -> Tuple[str, str]:
    """
    Parse folder name into config number and label.
    E.g., "conf_10_antiferro1" -> ("conf_10", "antiferro1")
    """
    parts = folder_name.split('_')
    if len(parts) >= 3:
        config = f"{parts[0]}_{parts[1]}"  # "conf_10"
        label = '_'.join(parts[2:])  # "antiferro1"
        return config, label
    return folder_name, ""


def group_by_config(rows: List[Dict]) -> Dict[str, List[Dict]]:
    """Group rows by config number."""
    groups = {}
    for row in rows:
        folder = row['Folder']
        config, label = parse_folder_name(folder)
        if config not in groups:
            groups[config] = []
        groups[config].append(row)
    return groups


def find_lowest_force_row(rows: List[Dict]) -> Dict:
    """Find the row with the lowest F_final value (excluding empty/NaN values)."""
    valid_rows = [
        row for row in rows 
        if row['F_final'].strip()  # Has F_final value
    ]
    
    if not valid_rows:
        # If no valid F_final, use the first row
        return rows[0]
    
    # Find row with minimum F_final
    return min(valid_rows, key=lambda r: float(r['F_final']))


def copy_files(
    source_folder: Path,
    dest_folders: List[Path],
    file_extensions: List[str] = ['.xyz', '.XV', '.xv']
) -> None:
    """Copy files with given extensions from source to destination folders."""
    # Find files with matching extensions in source folder
    source_files = []
    for ext in file_extensions:
        source_files.extend(source_folder.glob(f'*{ext}'))
    
    if not source_files:
        print(f"[!] No files found in {source_folder} with extensions {file_extensions}")
        return
    
    for dest_folder in dest_folders:
        # Create destination folder if it doesn't exist
        dest_folder.mkdir(parents=True, exist_ok=True)
        
        for source_file in source_files:
            dest_file = dest_folder / source_file.name
            shutil.copy2(str(source_file), str(dest_file))
            print(f"    ✓ Copied {source_file.name} to {dest_folder.name}")


def main(
    csv_path: Path,
    runs_root: Path,
    output_dir: Path,
    labels: List[str],
    file_extensions: List[str] = ['.xyz', '.XV', '.xv']
):
    """
    Main function to copy lowest force files.
    
    Args:
        csv_path: Path to summary_results.csv
        runs_root: Root path containing the run folders
        output_dir: Output directory where files will be copied
        labels: List of labels to process (e.g., ['antiferro1', 'antiferro2', 'ferro', 'spin2'])
        file_extensions: File extensions to copy (default: ['.xyz', '.XV'])
    """
    csv_path = Path(csv_path)
    runs_root = Path(runs_root)
    output_dir = Path(output_dir)
    
    if not csv_path.exists():
        print(f"[!] CSV file not found: {csv_path}")
        return
    
    if not runs_root.exists():
        print(f"[!] Runs root directory not found: {runs_root}")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Reading CSV file: {csv_path}")
    rows = read_summary_csv(csv_path)
    print(f"   Found {len(rows)} rows")
    
    # Group by config
    groups = group_by_config(rows)
    print(f"   Grouped into {len(groups)} configurations\n")
    
    # Process each group
    for config, group_rows in sorted(groups.items()):
        print(f"📁 Processing {config}:")
        
        # Find lowest force configuration
        lowest_row = find_lowest_force_row(group_rows)
        lowest_folder = lowest_row['Folder']
        f_final = lowest_row.get('F_final', 'N/A')
        
        print(f"   Lowest force config: {lowest_folder} (F_final: {f_final})")
        
        # Path to source folder (where files are)
        source_folder = runs_root / lowest_folder
        
        if not source_folder.exists():
            print(f"   [!] Source folder not found: {source_folder}")
            continue
        
        # Build destination folders
        dest_folders: List[Path] = []
        if labels:
            for label in labels:
                folder_name = f"{config}_{label}"
                dest_folders.append(output_dir / folder_name)
        else:
            for row in group_rows:
                folder_name = row['Folder']
                dest_folders.append(output_dir / folder_name)
        
        print(f"   Copying to {len(dest_folders)} folders...")
        copy_files(source_folder, dest_folders, file_extensions)
        print()


if __name__ == "__main__":
    # Default parameters - modify these as needed
    csv_path = "visualisation/conformers/slow_lowT/summary_results.csv"
    runs_root = "Short-2OS/runs/conformers/slow_lowT"
    output_dir = "Short-2OS/runs/conformers/slow_lowT_lowest_forces"
    labels = ['antiferro', 'ferro', '4radical', "4radical_2"]
    
    # Allow command-line overrides
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    if len(sys.argv) > 2:
        runs_root = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    if len(sys.argv) > 4:
        labels = sys.argv[4].split(',')
    
    print("=" * 70)
    print("Copy Lowest Force Configuration Files")
    print("=" * 70)
    print(f"CSV Path: {csv_path}")
    print(f"Runs Root: {runs_root}")
    print(f"Output Dir: {output_dir}")
    print(f"Labels: {labels}")
    print()
    
    main(csv_path, runs_root, output_dir, labels)
    
    print("=" * 70)
    print("✅ Done!")
    print("=" * 70)
