import os
from pathlib import Path
from rdkit import Chem

HERE = Path(__file__).resolve().parent

# Configuration
filename = "junction_SM"  # Base name for output files
slow_fast = "fast"  # Slow or fast FDF generation
label_out = "" #"_4radicals"  # Optional label for output directory

template_fdf = HERE / 'templates_fdf' / f'template_{filename}_{slow_fast}.fdf'     # Path to your FDF template
conformers_sdf = HERE / 'sdf_files' / f'{filename}.sdf' # Multi‐conformer SDF
output_dir = HERE / 'fdf_files' / filename / f'{slow_fast}{label_out}'     # Directory to save new FDFs
os.makedirs(output_dir, exist_ok=True)

# Dimension bounds (Angstrom)
x_bound, y_bound, z_bound = 25.0, 20.0, 20.0

# Read the template
with open(template_fdf) as f:
    lines = f.readlines()

# Locate coordinate block
coords_start = next(i for i, l in enumerate(lines) if l.strip().startswith('%block AtomicCoordinatesAndAtomicSpecies'))
coords_end   = next(i for i, l in enumerate(lines) if l.strip().startswith('%endblock AtomicCoordinatesAndAtomicSpecies'))

# Locate spin block
spin_start = next(i for i, l in enumerate(lines) if l.strip().startswith('%block DM.InitSpin'))
spin_end   = next(i for i, l in enumerate(lines) if l.strip().startswith('%endblock DM.InitSpin'))

# Split into sections
pre_coords    = lines[:coords_start+1]                                # up to and including block header
post_coords   = lines[coords_end:spin_start]                          # end coords block up to spin block start
post_spin     = lines[spin_end+1:]                                    # after spin block

# Atom type mapping
atomic_to_species = {6: 1, 1: 2}  # C:1, H:2

# Define spin variants
spin_variants = {
    'ferro': [
        "# --\n",
        "# SPINS\n",
        "# --\n",
        "Spin  polarized\n",
        "%block DM.InitSpin\n",
        "   29  1.0\n",
        "   42  1.0\n",
        "%endblock DM.InitSpin\n"
    ],
    'antiferro': [
        "# --\n",
        "# SPINS\n",
        "# --\n",
        "Spin  polarized\n",
        "%block DM.InitSpin\n",
        "   29  1.0\n",
        "   42 -1.0\n",
        "%endblock DM.InitSpin\n"
    ],
    '4radical': [
        "# --\n",
        "# SPINS\n",
        "# --\n",
        "Spin  polarized\n",
        "%block DM.InitSpin\n",
        "   3  1.0\n", # 3 for main config, 23 for conformers
        "   18  -1.0\n", # 18 for main config, 7 for conformers
        "   29  1.0\n",
        "   42 -1.0\n",
        "%endblock DM.InitSpin\n"
    ],
    '4radical_2': [
        "# --\n",
        "# SPINS\n",
        "# --\n",
        "Spin  polarized\n",
        "%block DM.InitSpin\n",
        "   3  1.0\n", # 3 for main config, 23 for conformers
        "   18   1.0\n", # 18 for main config, 7 for conformers 
        "   29 -1.0\n",
        "   42 -1.0\n",
        "%endblock DM.InitSpin\n"
    ]
}

# Load conformers
suppl = Chem.SDMolSupplier(conformers_sdf, removeHs=False)

# Generate FDFs
for idx, mol in enumerate(suppl, start=1):
    if mol is None:
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

print(f"Generated {idx * len(spin_variants)} FDF files in '{output_dir}'.")
