import os
import re
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Mesh_cutoff_sweep_conf_4_antiferro_with_DM")
print(path)
os.chdir(path)

# Regex patterns
free_energy_pattern = re.compile(r"Total\s*=\s*(-?\d+\.\d+)")
cutoff_pattern = re.compile(r"MeshCutoff\s+(\d+(?:\.\d+)?)\s+Ry")

results = {}

for folder in os.listdir():
    if os.path.isdir(folder) and folder.startswith("run_") and folder.endswith("Ry"):
        try:
            nn = int(folder[4:-2])  # extract integer from "run_{nn}Ry"
        except ValueError:
            continue  # skip folders not matching the expected pattern

        out_filename = f"siesta_{nn}Ry.out"
        output_path = os.path.join(folder, out_filename)

        if os.path.isfile(output_path):
            free_energy = None
            mesh_cutoff = None
            with open(output_path, "r") as f:
                for line in f:
                    if free_energy is None:
                        fe_match = free_energy_pattern.search(line)
                        if fe_match:
                            free_energy = float(fe_match.group(1))
                    if mesh_cutoff is None:
                        mc_match = cutoff_pattern.search(line)
                        if mc_match:
                            mesh_cutoff = float(mc_match.group(1))
                    if free_energy is not None and mesh_cutoff is not None:
                        break
            if free_energy is not None and mesh_cutoff is not None:
                results[folder] = (mesh_cutoff, free_energy)

# Print and plot results
for folder, (cutoff, energy) in sorted(results.items()):
    print(f"{folder}: MeshCutoff = {cutoff:.1f} Ry, Total Energy = {energy:.6f}")
    
cutoffs = [cutoff for cutoff, _ in results.values()]
energies = [energy for _, energy in results.values()]

cutoffs, energies = zip(*sorted(zip(cutoffs, energies)))
dE = np.array(energies) - energies[-1]

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(cutoffs, dE, marker='o', linestyle='-', color='b')
ax.set_xlabel('Mesh Cutoff (Ry)')
ax.set_ylabel('Error Total Energy (eV)')
# ax.set_ylim(-8851.625, -8851.5)
# ax.set_xlim(0, 700)
# ax.set_title('Total Energy vs Mesh Cutoff')
# ax.grid(True)

# ------------------------ Inset: dE/dCutoff --------------------------------
# Finite‑difference derivative (uses central differences internally)
dE = np.array(energies) - energies[-1]

# Create an inset occupying ~40% of the parent axes, in the upper‑right corner
in_ax = inset_axes(ax, width="50%", height="50%", loc='upper right', borderpad=1)

in_ax.plot(cutoffs, dE, marker='s', linestyle='--')
# in_ax.set_xlabel('Cutoff (Ry)', fontsize=7)
# in_ax.set_ylabel('dE (eV)', fontsize=7)
in_ax.tick_params(axis='both', which='major', labelsize=7)
in_ax.set_xlim(300, 800)
in_ax.set_ylim(0, 0.01)
in_ax.grid(True)

fig.tight_layout()
fig.savefig("FreeEnergy_vs_MeshCutoff_with_derivative.png", dpi=300)
# plt.show()
