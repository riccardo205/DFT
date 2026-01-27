from rdkit import Chem
from rdkit.Chem import AllChem

# 1) Load your molecule
mol = Chem.MolFromMolFile('2OS-Short_init0.mol', removeHs=False)
if mol is None:
    raise ValueError("Failed to read molecule file.")

Chem.SanitizeMol(mol)  # will assign aromaticity

# ensure explicit hydrogens
# mol = Chem.AddHs(mol)

# 2) Embed multiple conformers
num_confs = 20
params = AllChem.ETKDGv3()
params.numThreads = 0        # use all CPUs
params.pruneRmsThresh = 0.5  # discard near-duplicates (RMSD < 0.5 Å)
ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)

# 3) Energy-minimize each conformer (UFF)
# for cid in ids:
#     AllChem.UFFOptimizeMolecule(mol, confId=cid)

# 4) Write them all to a single SDF
w = Chem.SDWriter('conformers_not_optimised.sdf')
for cid in ids:
    w.write(mol, confId=cid)
w.close()

print(f"Generated {len(ids)} conformers → conformers.sdf")
