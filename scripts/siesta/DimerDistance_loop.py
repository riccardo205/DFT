import os
import re
import shutil
import subprocess
import numpy as np

# Define variables
folder_path = "../dimer/"
siesta_executable = "siesta"
input_template = "dimer_distance.fdf"  # Your Siesta input template file
pseudo_file = "C.psf"  # Update with your pseudopotential filename

os.chdir(folder_path)
print("Current directory:", os.getcwd())

# Create a csv file where to store the result
out_folder = input("Insert filename for final results (without extension):")
output_csv = out_folder + f"/{out_folder}.csv"

bohr_distance = np.arange(1, 4, 0.02)  # Modify as needed

if os.path.exists(output_csv):
    print(f"{output_csv} already exists. Not creating a new file.")
else:
    os.makedirs(out_folder, exist_ok=True)
    with open(output_csv, "w") as csv_file:
        csv_file.write("Run,Interatomic Distance (Bohr),Interatomic Distance (Å),Total Energy (eV)\n")
        

# Loop over different lattice constants
for i, distance in enumerate(bohr_distance):
    run_folder = f"{out_folder}/runs/run{i:03}"
    
    # Create a new directory for the run
    os.makedirs(run_folder, exist_ok=True)

    # Copy pseudopotential file into the run folder
    shutil.copy(pseudo_file, run_folder)

    # Modify input file
    with open(input_template, "r") as f:
        input_data = f.read()
    
    modified_input = input_data.replace("%DISTANCE_DIMER%", f"{distance:.3f}")

    # Save modified input in run folder
    input_file_path = os.path.join(run_folder, input_template)
    with open(input_file_path, "w") as f:
        f.write(modified_input)

    # Run Siesta inside the run folder
    command = f"{siesta_executable} < {input_template} |tee siesta.out"
    subprocess.run(command, shell=True, executable="/bin/bash", cwd=run_folder)

    # Extract total energy from output
    siesta_output = os.path.join(run_folder, "siesta.out")
    with open(siesta_output, "r") as f:
        content = f.read()
    pattern = r"Total\s*=\s*([-+]?\d*\.\d+)"
    matches = re.findall(pattern, content)
    total_energy = matches[0] or "N/A"
    
    # Find interatomic distance
    bond_final = os.path.join(run_folder, input_template[:-4]+".BONDS_FINAL")
    with open(bond_final, 'r') as file:
        content = file.read()
    
    # Use a regex to find all occurrences of the bond distance.
    # The pattern looks for a floating-point number followed by "Ang. Really at:"
    pattern = r'\s+([0-9]+\.[0-9]+)\s+Ang\. Really at:'
    matches = re.findall(pattern, content)
    
    # Convert matches to float values
    bond_distances = [float(distance) for distance in matches]
    print("Extracted bond distances:", bond_distances)
    bond_distance = bond_distances[0] if bond_distances else "N/A"
    bond_distance_bohr = bond_distance / 0.529177 if bond_distances!= "N/A" else "N/A"
    
    # Store results
    with open(output_csv, "a") as csv_file:
        csv_file.write(f"{i},{bond_distance_bohr:.3f},{bond_distance},{total_energy}\n")
    
    print(f"Run {i} completed. Interatomic Distance: {bond_distance} Å, Energy: {total_energy} eV")

print("All simulations completed. Results saved in", output_csv)
