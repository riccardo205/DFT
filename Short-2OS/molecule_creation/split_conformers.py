from rdkit import Chem
import os

filename = 'conformers_not_optimised'
folder = filename

# Ensure the folder exists
if not os.path.exists(folder):
    os.makedirs(folder)

# 1) Read all conformers from the multi‐conformer SDF
suppl = Chem.SDMolSupplier(f'{filename}.sdf', removeHs=False)

# 2) Loop and write each out
for idx, mol in enumerate(suppl, start=1):
    if mol is None:
        continue
    
    path = os.path.join(folder, f'conf_{idx}')
    # a) SDF
    writer = Chem.SDWriter(f'{path}.sdf')
    writer.write(mol)
    writer.close()
    
    # b) MOL (Avogadro-friendly)
    Chem.MolToMolFile(mol, f'{path}.mol')
    
    # c) XYZ
    xyz_block = Chem.MolToXYZBlock(mol)
    with open(f'{path}.xyz', 'w') as f:
        f.write(xyz_block)

print(f"Saved {idx} conformers as conf_1–conf_{idx} in .sdf, .mol, and .xyz formats.")