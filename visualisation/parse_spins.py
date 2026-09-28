#!/usr/bin/env python3
"""
Script to parse spin data from all_spins.csv and separate atoms by spin component sign.
For a given folder and spin component, creates two lists:
- Atoms with positive spin values
- Atoms with negative spin values
"""

import csv
import sys
from pathlib import Path


def parse_spin_data(csv_file, folder_name, spin_component, group_boundaries=None, output_file=None):
    """
    Parse spin data from CSV file and separate atoms by component sign.
    
    Args:
        csv_file (str): Path to the all_spins.csv file
        folder_name (str): Name of the folder/configuration to extract (e.g., 'conf_11_antiferro1')
        spin_component (str): Spin component to analyze ('Sx', 'Sy', or 'Sz')
        group_boundaries (list): List of atom indices to define group boundaries (e.g., [86] for atoms 1-86, 87-end)
                                If None, treats all atoms as one group
        output_file (str): Path to save the output. If None, uses default naming
    
    Returns:
        tuple: If no groups: (positive_atoms, negative_atoms)
               If groups: list of tuples [(pos_group1, neg_group1), (pos_group2, neg_group2), ...]
    """
    
    # Map spin components to CSV column names
    column_map = {
        'Sx': 'Spin_final_all_Sx',
        'Sy': 'Spin_final_all_Sy',
        'Sz': 'Spin_final_all_Sz',
    }
    
    if spin_component not in column_map:
        raise ValueError(f"Invalid spin component: {spin_component}. Must be 'Sx', 'Sy', or 'Sz'")
    
    target_column = column_map[spin_component]
    all_atoms = {}  # Dictionary to store all atom data: {atom_num: value}
    
    # Read CSV file
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            if row['Folder'] == folder_name:
                spin_data = row[target_column].strip()
                
                if not spin_data:
                    print(f"Warning: No spin data found for {folder_name} component {spin_component}")
                    if group_boundaries:
                        return [([], [])] * len(group_boundaries)
                    return [], []
                
                # Parse the spin data format: "1:-0.004; 2:0.002; 3:-0.002; ..."
                pairs = spin_data.split(';')
                
                for pair in pairs:
                    pair = pair.strip()
                    if not pair:
                        continue
                    
                    try:
                        atom_num_str, value_str = pair.split(':')
                        atom_num = int(atom_num_str.strip())
                        value = float(value_str.strip())
                        all_atoms[atom_num] = value
                    
                    except ValueError:
                        print(f"Warning: Could not parse pair '{pair}'")
                        continue
                
                # Found the folder, now process groups if specified
                if group_boundaries:
                    # Create group ranges
                    boundaries = [0] + sorted(group_boundaries)
                    max_atom = max(all_atoms.keys()) if all_atoms else 0
                    
                    results = []
                    for i in range(len(boundaries)):
                        start = boundaries[i] + 1
                        end = boundaries[i + 1] if i + 1 < len(boundaries) else max_atom
                        
                        positive_atoms = []
                        negative_atoms = []
                        
                        for atom_num in range(start, end + 1):
                            if atom_num in all_atoms:
                                value = all_atoms[atom_num]
                                if value > 0:
                                    positive_atoms.append(atom_num)
                                elif value < 0:
                                    negative_atoms.append(atom_num)
                        
                        results.append((sorted(positive_atoms), sorted(negative_atoms)))
                    
                    return results
                else:
                    # No groups, return single positive and negative lists
                    positive_atoms = []
                    negative_atoms = []
                    
                    for atom_num, value in all_atoms.items():
                        if value > 0:
                            positive_atoms.append(atom_num)
                        elif value < 0:
                            negative_atoms.append(atom_num)
                    
                    return sorted(positive_atoms), sorted(negative_atoms)
    
    print(f"Error: Folder '{folder_name}' not found in {csv_file}")
    if group_boundaries:
        return [([], [])] * len(group_boundaries)
    return [], []


def save_results(results, output_file, group_boundaries=None):
    """
    Save the separated atom lists to a text file.
    
    Args:
        results: If no groups: tuple of (positive_atoms, negative_atoms)
                If groups: list of tuples [(pos_group1, neg_group1), ...]
        output_file (str): Path to output file
        group_boundaries (list): Boundaries used for grouping (for display purposes)
    """
    with open(output_file, 'w') as f:
        if group_boundaries:
            # Multiple groups
            f.write(f"GROUPED ANALYSIS - Group boundaries: {group_boundaries}\n")
            f.write("=" * 70 + "\n\n")
            
            boundaries = [0] + sorted(group_boundaries)
            
            for i, (positive_atoms, negative_atoms) in enumerate(results):
                start = boundaries[i] + 1
                end = boundaries[i + 1] if i + 1 < len(boundaries) else "end"
                
                f.write(f"GROUP {i + 1} (atoms {start}-{end}):\n")
                f.write("-" * 70 + "\n")
                
                f.write("POSITIVE ATOMS (spin component > 0):\n")
                f.write(str(positive_atoms) + "\n")
                f.write(f"Count: {len(positive_atoms)}\n\n")
                
                f.write("NEGATIVE ATOMS (spin component < 0):\n")
                f.write(str(negative_atoms) + "\n")
                f.write(f"Count: {len(negative_atoms)}\n")
                f.write("\n" + "=" * 70 + "\n\n")
        else:
            # Single group (all atoms)
            positive_atoms, negative_atoms = results
            
            f.write("POSITIVE ATOMS (spin component > 0):\n")
            f.write(str(positive_atoms) + "\n")
            f.write(f"Count: {len(positive_atoms)}\n")
            f.write("\n")
            
            f.write("NEGATIVE ATOMS (spin component < 0):\n")
            f.write(str(negative_atoms) + "\n")
            f.write(f"Count: {len(negative_atoms)}\n")
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    # Configuration - modify these variables or use command-line arguments
    CSV_FILE = "visualisation/output/dimers_run2/non_colinear_fast2/all_spins.csv"
    FOLDER_NAME = "conf_11_antiferro1"
    SPIN_COMPONENT = "Sz"  # Can be 'Sx', 'Sy', or 'Sz'
    GROUP_BOUNDARIES = [86]  # e.g., [86] to split into atoms 1-86 and 87-end, or [50, 100] for 1-50, 51-100, 101-end
    
    # Allow command-line argument override
    if len(sys.argv) > 1:
        CSV_FILE = sys.argv[1]
    if len(sys.argv) > 2:
        FOLDER_NAME = sys.argv[2]
    if len(sys.argv) > 3:
        SPIN_COMPONENT = sys.argv[3]
    if len(sys.argv) > 4:
        # Parse group boundaries from comma-separated string, e.g., "86" or "86,100"
        GROUP_BOUNDARIES = [int(x.strip()) for x in sys.argv[4].split(',')]
    
    # Save output file in the same directory as CSV_FILE
    csv_path = Path(CSV_FILE)
    group_suffix = f"_groups{'_'.join(map(str, GROUP_BOUNDARIES))}" if GROUP_BOUNDARIES else ""
    OUTPUT_FILE = str(csv_path.parent / f"{FOLDER_NAME}_{SPIN_COMPONENT}{group_suffix}_separated_atoms.txt")
    
    # Allow command-line argument override for output file
    if len(sys.argv) > 5:
        OUTPUT_FILE = sys.argv[5]
    
    print(f"Parsing spin data:")
    print(f"  CSV File: {CSV_FILE}")
    print(f"  Folder: {FOLDER_NAME}")
    print(f"  Component: {SPIN_COMPONENT}")
    if GROUP_BOUNDARIES:
        print(f"  Group boundaries: {GROUP_BOUNDARIES}")
    print(f"  Output: {OUTPUT_FILE}\n")
    
    results = parse_spin_data(CSV_FILE, FOLDER_NAME, SPIN_COMPONENT, GROUP_BOUNDARIES)
    
    if GROUP_BOUNDARIES:
        print(f"Results for {len(results)} groups:")
        boundaries = [0] + sorted(GROUP_BOUNDARIES)
        for i, (positive, negative) in enumerate(results):
            start = boundaries[i] + 1
            end = boundaries[i + 1] if i + 1 < len(boundaries) else "end"
            print(f"  Group {i + 1} (atoms {start}-{end}):")
            print(f"    Positive ({len(positive)}): {positive}")
            print(f"    Negative ({len(negative)}): {negative}")
        print()
    else:
        positive, negative = results
        print(f"Positive atoms ({len(positive)}): {positive}")
        print(f"Negative atoms ({len(negative)}): {negative}\n")
    
    save_results(results, OUTPUT_FILE, GROUP_BOUNDARIES)
