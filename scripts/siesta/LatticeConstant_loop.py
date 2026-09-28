import os
import re
import shutil
import subprocess
import numpy as np

# Define variables
folder_path = "Al_loops/"
siesta_executable = "siesta"  # Update with the correct path
input_template = "Al_bulk.fdf"  # Your Siesta input template file
pseudo_file = "Al.psf"  # Update with your pseudopotential filename

print("Current directory:", os.getcwd())
os.chdir(folder_path)
print("Current directory:", os.getcwd())

output_csv = input("Insert filename for final results (without extension):") + ".csv"
lattice_constants = np.arange(3.8, 4.2, 0.01)  # Modify as needed

# Create/open results file
with open(output_csv, "w") as csv_file:
    csv_file.write("Run,Lattice Constant (Å),Total Energy (eV)\n")

# Loop over different lattice constants
for i, lattice in enumerate(lattice_constants):
    run_folder = f"runs/run{i}"
    
    # Create a new directory for the run
    os.makedirs(run_folder, exist_ok=True)

    # Copy pseudopotential file into the run folder
    shutil.copy(pseudo_file, run_folder)

    # Modify input file
    with open(input_template, "r") as f:
        input_data = f.read()
    
    modified_input = input_data.replace("%LATTICE_CONSTANT%", str(lattice))

    # Save modified input in run folder
    input_file_path = os.path.join(run_folder, "Al_bulk.fdf")
    with open(input_file_path, "w") as f:
        f.write(modified_input)

    # Run Siesta inside the run folder
    command = f"{siesta_executable} < {input_template} |tee siesta.out"
    subprocess.run(command, shell=True, executable="/bin/bash", cwd=run_folder)

    # Extract total energy from output
    siesta_output = os.path.join(run_folder, "siesta.out")
    total_energy = "N/A"
    with open(siesta_output, "r") as f:
        for line in f:
            match = re.search(r"Total\s*=\s*([-+]?\d*\.\d+)", line)
            if match:
                total_energy = match.group(1)  # Extract energy value
                break

    # Store results
    with open(output_csv, "a") as csv_file:
        csv_file.write(f"{i},{lattice},{total_energy}\n")

    print(f"Run {i} completed. Lattice constant: {lattice} Å, Energy: {total_energy} eV")

print("All simulations completed. Results saved in", output_csv)
