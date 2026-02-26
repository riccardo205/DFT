import os
from pathlib import Path
from rdkit import Chem

HERE = Path(__file__).resolve().parent

# Configuration
filename = "junction_SM2"  # Base name for output files
slow_fast = "fast"  # Slow or fast FDF generation
label_out = "" #"_4radicals"  # Optional label for output directory

template_fdf = HERE / 'templates_fdf' / f'template_{filename}_{slow_fast}.fdf'     # Path to your FDF template
conformers_sdf = HERE / 'sdf_files' / f'{filename}.sdf' # Multi‐conformer SDF
output_dir = HERE / 'fdf_files' / filename / f'{slow_fast}{label_out}'     # Directory to save new FDFs
os.makedirs(output_dir, exist_ok=True)

# Dimension bounds (Angstrom)
x_bound, y_bound, z_bound = 45.0, 35.0, 25.0

# Read the template
with open(template_fdf) as f:
    lines = f.readlines()

# Locate coordinate block
coords_start = next(i for i, l in enumerate(lines) if l.strip().startswith('%block AtomicCoordinatesAndAtomicSpecies'))
coords_end   = next(i for i, l in enumerate(lines) if l.strip().startswith('%endblock AtomicCoordinatesAndAtomicSpecies'))

# Locate spin block (if present)
try:
    spin_start = next(i for i, l in enumerate(lines) if l.strip().startswith('%block DM.InitSpin'))
    spin_end = next(i for i, l in enumerate(lines) if l.strip().startswith('%endblock DM.InitSpin'))
    has_spin_block = True
except StopIteration:
    spin_start = None
    spin_end = None
    has_spin_block = False

# Split into sections
pre_coords = lines[:coords_start+1]                                   # up to and including block header
if has_spin_block:
    post_coords = lines[coords_end:spin_start]                        # end coords block up to spin block start
    post_spin = lines[spin_end+1:]                                    # after spin block
else:
    post_coords = lines[coords_end:]                                  # end coords block to end of file
    post_spin = []

# Atom type mapping
atomic_to_species = {6: 1, 1: 2, 79: 3}  # C:1, H:2, Au:3

# Define spin variants
# spin_variants = {
#     'ferro': [
#         "# --\n",
#         "# SPINS\n",
#         "# --\n",
#         "Spin  polarized\n",
#         "%block DM.InitSpin\n",
#         "   29  1.0\n",
#         "   42  1.0\n",
#         "%endblock DM.InitSpin\n"
#     ],
#     'antiferro': [
#         "# --\n",
#         "# SPINS\n",
#         "# --\n",
#         "Spin  polarized\n",
#         "%block DM.InitSpin\n",
#         "   29  1.0\n",
#         "   42 -1.0\n",
#         "%endblock DM.InitSpin\n"
#     ],
#     '4radical': [
#         "# --\n",
#         "# SPINS\n",
#         "# --\n",
#         "Spin  polarized\n",
#         "%block DM.InitSpin\n",
#         "   3  1.0\n", # 3 for main config, 23 for conformers
#         "   18  -1.0\n", # 18 for main config, 7 for conformers
#         "   29  1.0\n",
#         "   42 -1.0\n",
#         "%endblock DM.InitSpin\n"
#     ],
#     '4radical_2': [
#         "# --\n",
#         "# SPINS\n",
#         "# --\n",
#         "Spin  polarized\n",
#         "%block DM.InitSpin\n",
#         "   3  1.0\n", # 3 for main config, 23 for conformers
#         "   18   1.0\n", # 18 for main config, 7 for conformers 
#         "   29 -1.0\n",
#         "   42 -1.0\n",
#         "%endblock DM.InitSpin\n"
#     ]
# }
spin_variants = {'antiferro': [
        "# --\n",
        "# SPINS\n",
        "# --\n",
        "Spin  polarized\n",
        "%block DM.InitSpin\n",
        "   29  1.0\n",
        "   42 -1.0\n",
        "%endblock DM.InitSpin\n"
    ],
}
# spin_variants = {}  # No spin variants; just one FDF per conformer

# Load conformers
suppl = Chem.SDMolSupplier(conformers_sdf, removeHs=False)

generated_count = 0
warned_missing_spin_block = False

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

    # Write files based on spin variants; if none, write a single file without spin block.
    if spin_variants and has_spin_block:
        for variant, spin_block in spin_variants.items():
            fname = os.path.join(output_dir, f'conf_{idx}_{variant}.fdf')
            with open(fname, 'w') as f:
                f.writelines(pre_coords)
                f.writelines(coords_lines)
                f.writelines(post_coords)
                f.writelines(spin_block)
                f.writelines(post_spin)
            generated_count += 1
    else:
        if spin_variants and not has_spin_block and not warned_missing_spin_block:
            print("Warning: spin variants requested but no %block DM.InitSpin found in template; skipping spin insertion.")
            warned_missing_spin_block = True
        fname = os.path.join(output_dir, f'conf_{idx}.fdf')
        with open(fname, 'w') as f:
            f.writelines(pre_coords)
            f.writelines(coords_lines)
            f.writelines(post_coords)
            f.writelines(post_spin)
        generated_count += 1

print(f"Generated {generated_count} FDF files in '{output_dir}'.")
