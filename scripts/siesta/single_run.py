import os
import re
import shutil
import subprocess
import numpy as np

# Define variables
folder_path = "../Short-2OS/"
siesta_executable = "siesta"
input_template = "short-2OS.fdf"  # Your Siesta input template file

os.chdir(folder_path)
print("Current directory:", os.getcwd())

# Create a csv file where to store the result
out_folder = input("Insert filename for final results (without extension):")
output_csv = out_folder + f"/{out_folder}.csv"

if os.path.exists(output_csv):
    print(f"{output_csv} already exists. Not creating a new file.")
else:
    os.makedirs(out_folder)
    with open(output_csv, "w") as csv_file:
        csv_file.write("Run,Total Energy (eV)\n")

# Find the correct run number checking the previous runs
i = 0
while os.path.exists(f"{out_folder}/runs/run{i:03}"):
    i += 1
run_folder = f"{out_folder}/runs/run{i:03}"

# Create a new directory for the run
os.makedirs(run_folder, exist_ok=True)

# Copy pseudopotential and input files into the run folder
for pseudo_file in os.listdir("pseudopotentials"):
    pseudopotential = os.path.join("pseudopotentials", pseudo_file)
    shutil.copy(pseudopotential, run_folder)
shutil.copy(input_template, run_folder)

# Run Siesta inside the run folder
command = f"{siesta_executable} < {input_template} |tee siesta.out"
subprocess.run(command, shell=True, executable="/bin/bash", cwd=run_folder)

# Extract total energy from output
siesta_output = os.path.join(run_folder, "siesta.out")
with open(siesta_output, "r") as f:
    content = f.read()
pattern = r"Total\s*=\s*([-+]?\d*\.\d+)"
matches = re.findall(pattern, content)
total_energy = matches[0] if matches else "N/A"


# Store results
with open(output_csv, "a") as csv_file:
    csv_file.write(f"{i},{total_energy}\n")

print(f"Run {i} completed. Energy: {total_energy} eV")

