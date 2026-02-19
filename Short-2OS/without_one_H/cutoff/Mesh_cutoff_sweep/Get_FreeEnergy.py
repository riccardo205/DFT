import os
import re

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

# Print results
for folder, (cutoff, energy) in sorted(results.items()):
    print(f"{folder}: MeshCutoff = {cutoff:.1f} Ry, Total Energy = {energy:.6f}")
 

