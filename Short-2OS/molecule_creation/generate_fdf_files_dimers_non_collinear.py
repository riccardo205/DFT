import os
from pathlib import Path
from rdkit import Chem

HERE = Path(__file__).resolve().parent

# Configuration
filename = "dimers_run2"  # Base name for output files
slow_fast = "non_colinear"  # Slow or fast FDF generation

template_fdf = HERE / 'templates_fdf' / f'template_dimer_{slow_fast}.fdf'     # Path to your FDF template
conformers_sdf = HERE / 'sdf_files' / f'{filename}.sdf' # Multi‐conformer SDF
output_dir = HERE / 'fdf_files' / filename / slow_fast     # Directory to save new FDFs
os.makedirs(output_dir, exist_ok=True)

# Select specific conformer indices (set to None or empty set to process all)
selected_indices = {11}  # Example: {1, 3, 5, 7} to only save conformers 1, 3, 5, and 7
# selected_indices = {1, 2, 3}  # Uncomment and modify to select specific conformers

# Dimension bounds (Angstrom)
x_bound, y_bound, z_bound = 28.0, 23.0, 20.0

# Read the template
with open(template_fdf) as f:
    lines = f.readlines()

# Locate coordinate block
coords_start = next(i for i, l in enumerate(lines) if l.strip().startswith('%block AtomicCoordinatesAndAtomicSpecies'))
coords_end   = next(i for i, l in enumerate(lines) if l.strip().startswith('%endblock AtomicCoordinatesAndAtomicSpecies'))

# Locate spin block
spin_start = next(i for i, l in enumerate(lines) if l.strip().startswith('# SPINS'))
spin_end   = next(i for i, l in enumerate(lines) if l.strip().startswith('%endblock DM.InitSpin'))

# Split into sections
pre_coords    = lines[:coords_start+1]                                # up to and including block header
post_coords   = lines[coords_end:spin_start]                          # end coords block up to spin block start
post_spin     = lines[spin_end+1:]                                    # after spin block

# Atom type mapping
atomic_to_species = {6: 1, 1: 2}  # C:1, H:2

"""
"# --\n",
"# SPINS\n",
"# --\n",
Spin  non-colinear
%block DM.InitSpin
# ia   |S|   theta(deg)   phi(deg)
  29   1.0     0.0        0.0
  42   1.0     0.0        0.0
 115   1.0     0.0        0.0
 128   1.0     0.0        0.0
%endblock DM.InitSpin
"""
# Define spin variants
spin_variants = {
    'ferro': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     5.0        0.0\n",
        "   42  1.0     5.0        0.0\n",
        "   115 1.0     175.0        0.0\n",
        "   128 1.0     175.0        0.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'antiferro1': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     5.0        0.0\n",
        "   42  1.0     175.0        0.0\n",
        "   115 1.0     5.0        0.0\n",
        "   128 1.0     175.0        0.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'antiferro2': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     5.0        0.0\n",
        "   42  1.0     175.0        0.0\n",
        "   115 1.0     175.0        0.0\n",
        "   128 1.0     5.0        0.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'antiferro3': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     90.0        0.0\n",
        "   42  1.0     90.0        180.0\n",
        "   115 1.0     180.0        0.0\n",
        "   128 1.0     0.0        0.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'antiferro4': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     90.0        0.0\n",
        "   42  1.0     90.0        180.0\n",
        "   115 1.0     0.0        0.0\n",
        "   128 1.0     180.0        0.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'antiferro5': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     90.0        0.0\n",
        "   42  1.0     180.0        0.0\n",
        "   115 1.0     0.0        0.0\n",
        "   128 1.0     90.0        0.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'spin2': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     5.0        0.0\n",
        "   42  1.0     5.0        0.0\n",
        "   115 1.0     0.0        0.0\n",
        "   128 1.0     0.0        0.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'spin1_a': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     175.0        0.0\n",
        "   42  1.0     5.0        0.0\n",
        "   115 1.0     5.0        0.0\n",
        "   128 1.0     5.0        0.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'spin1_b': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     5.0        0.0\n",
        "   42  1.0     175.0        0.0\n",
        "   115 1.0     5.0        0.0\n",
        "   128 1.0     5.0        0.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'spin1_c': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     5.0        0.0\n",
        "   42  1.0     5.0        0.0\n",
        "   115 1.0     175.0        0.0\n",
        "   128 1.0     5.0        0.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'spin1_d': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     5.0        0.0\n",
        "   42  1.0     5.0        0.0\n",
        "   115 1.0     5.0        0.0\n",
        "   128 1.0     175.0        0.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'spin1_e': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     45.0        0.0\n",
        "   42  1.0     45.0        0.0\n",
        "   115 1.0     90.0        180.0\n",
        "   128 1.0     90.0        180.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'spin1_f': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     45.0        0.0\n",
        "   42  1.0     90.0        180.0\n",
        "   115 1.0     90.0        180.0\n",
        "   128 1.0     45.0        0.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'planar_xy': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     90.0        0.0\n",
        "   42  1.0     90.0        90.0\n",
        "   115 1.0     90.0        180.0\n",
        "   128 1.0     90.0        270.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'canted_45': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     45.0        0.0\n",
        "   42  1.0     45.0        180.0\n",
        "   115 1.0     135.0        0.0\n",
        "   128 1.0     135.0        180.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'tetrahedral': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     54.7        0.0\n",
        "   42  1.0     54.7        120.0\n",
        "   115 1.0     125.3        60.0\n",
        "   128 1.0     125.3        180.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'spiral_120': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     90.0        0.0\n",
        "   42  1.0     90.0        120.0\n",
        "   115 1.0     90.0        240.0\n",
        "   128 1.0     90.0        0.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'mixed_135': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     0.0        0.0\n",
        "   42  1.0     135.0        0.0\n",
        "   115 1.0     135.0        180.0\n",
        "   128 1.0     180.0        0.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'perpendicular': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     90.0        0.0\n",
        "   42  1.0     90.0        0.0\n",
        "   115 1.0     90.0        0.0\n",
        "   128 1.0     90.0        0.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'canted_60': [
        "# SPINS\n",
        "# --\n",
        "Spin  non-colinear\n",
        "%block DM.InitSpin\n",
        "   29  1.0     60.0        0.0\n",
        "   42  1.0     60.0        180.0\n",
        "   115 1.0     120.0        0.0\n",
        "   128 1.0     120.0        180.0\n",
        "%endblock DM.InitSpin\n"
    ],
}

# Load conformers
suppl = Chem.SDMolSupplier(conformers_sdf, removeHs=False)

# Generate FDFs
generated_count = 0
for idx, mol in enumerate(suppl, start=1):
    if mol is None:
        continue
    
    # Check if this conformer should be processed
    if selected_indices is not None and idx not in selected_indices:
        continue

    # Gather atomic positions
    conf = mol.GetConformer()
    positions = [conf.GetAtomPosition(a.GetIdx()) for a in mol.GetAtoms()]
    xs = [p.x for p in positions]; ys = [p.y for p in positions]; zs = [p.z for p in positions]
    x_spread = max(xs) - min(xs)
    y_spread = max(ys) - min(ys)
    z_spread = max(zs) - min(zs)

    # Check bounds
    if x_spread > x_bound or y_spread > y_bound or z_spread > z_bound:
        print(f"Skipping conformer {idx}: spread (x,y,z)=({x_spread:.2f},{y_spread:.2f},{z_spread:.2f}) exceeds bounds ({x_bound},{y_bound},{z_bound})")
        continue

    # Build coordinate lines
    coords_lines = [
        f"  {p.x:12.6f}  {p.y:12.6f}  {p.z:12.6f}        {atomic_to_species[a.GetAtomicNum()]}\n"
        for a, p in zip(mol.GetAtoms(), positions)
    ]

    # For each spin variant, write a file
    for variant, spin_block in spin_variants.items():
        fname = os.path.join(output_dir, f'conf_{idx}_{variant}.fdf')
        with open(fname, 'w') as f:
            f.writelines(pre_coords)
            f.writelines(coords_lines)
            f.writelines(post_coords)
            f.writelines(spin_block)
            f.writelines(post_spin)
    
    generated_count += 1

print(f"Generated {generated_count * len(spin_variants)} FDF files in '{output_dir}'.")
