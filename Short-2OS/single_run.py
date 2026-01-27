import os
import re
import shutil
import subprocess

# Define variables
siesta_executable = "siesta5"  # Update with the correct path
input_template = "fdf_files/short-2OS_init0_ferro.fdf"  # Your Siesta input template file
pseudo_files = ["H.psf", "C.psf"]
batch_file = "siesta5_relaxation.sh"

#folder for storing output runs
label_folder = input("You can choose a label for the run. [""] ") or ""
if label_folder != "":
    out_folder = "runs/" + label_folder + "/run/"
else:
    out_folder = "runs/run"

ii = 0
if os.path.exists(out_folder):
    ow_flag = input("Output folder already exists. Overwrite? A new folder with incremental number will be created otherwise. y/[n]  ") or "n"
    if ow_flag is "n":
        tmp_folder = out_folder + f"_{ii:03}"
        while os.path.exists(tmp_folder):
            ii += 1
            tmp_folder = out_folder + f"_{ii:03}"
        out_folder = tmp_folder
        os.makedirs(out_folder, exist_ok=True)

print(f"\n================================================================================")
print(f"Running {siesta_executable} simulations from input template:  {input_template}\n")
print(f"Jobs batched starting from batch file:  {batch_file}\n")
print(f"Storing output data in {out_folder}\n")

batch_pars = ["01:00:00", "16", "1", "2G"]
default_flag  = input("Do you want to use default parameters for jobs? [y]/n  ") or "y"
if default_flag is not "y":
    # Ask wall time
    pattern = re.compile(r"^(?:[0-1]\d|2[0-3]):[0-5]\d:[0-5]\d$")  # matches 0–23:0–59:0–59
    ask_flag = True
    while ask_flag:
        batch_pars[0] = input("Input desired maximum wall time (default 01:00:00):  ") or batch_pars[0]
        if pattern.match(batch_pars[0]):
            ask_flag = False
        else:
            print("Invalid format! Insert time as hh:mm:ss!")

    # Ask number of nodes
    ask_flag = True
    while ask_flag:
        try:
            Ntasks = int(input("Input desired number of tasks (default 16):  ") or "16")
            if Ntasks > 0:
                ask_flag = False
                batch_pars[1] = str(Ntasks)
            else:
                print("Value must be greater than 0!")
        except:
            print("Invalid input. Please enter an integer greater than 0!")

    # Ask number of tasks per node
    ask_flag = True
    while ask_flag:
        try:
            CperT = int(input("Input desired number of cpu per task (default 1):  ") or "1")
            if CperT > 0:
                ask_flag = False
                batch_pars[2] = str(CperT)
            else:
                print("Value must be greater than 0!")
        except:
            print("Invalid input. Please enter an integer greater than 0!")

    # Ask RAM quantity per cpu
    ask_flag = True
    while ask_flag:
        try:
            memory = int(input("Input desired RAM per cpu, in GBytes (default 2):  ") or "2")
            if memory > 0:
                if memory > 3.9:
                    cut_flag = input("Requested RAM exceeds limit of 3968M per cpu. Cut value to 3968M? [y]/n  ") or "y"
                    if cut_flag is "y":
                        ask_flag = False
                        batch_pars[3] = "3968M"
                else:
                    ask_flag = False
                    batch_pars[3] = str(memory) + "G"
            else:
                print("Value must be greater than 0!")
        except:
            print("Invalid input. Please enter an integer greater than 0!")
            

run_folder = out_folder

if not os.path.exists(run_folder):
    # Create a new directory for the run
    os.makedirs(run_folder, exist_ok=True)

# Copy pseudopotential and input files into the run folder
for file in pseudo_files:
    shutil.copy(os.path.join("pseudopotentials",file), run_folder)

# Copy input file
shutil.copy(input_template, run_folder)

# Modify batch file
with open(batch_file, "r") as f:
    batch_lines = f.read()
# make sure that the default input has these strings
batch_lines = batch_lines.replace(r"%EXECUTABLE%", f"\"{siesta_executable}\"")  #job name
batch_lines = batch_lines.replace(r"%OUTPUTFILE%", f"siesta.out")  #output file name
batch_lines = batch_lines.replace(r"%ERRORFILE%", f"siesta.err")  #error file name
batch_lines = batch_lines.replace(r"%WALLTIME%", batch_pars[0])  #maximum wall time for computation
batch_lines = batch_lines.replace(r"%NTASKS%", batch_pars[1])  #number of nodes to use
batch_lines = batch_lines.replace(r"%CPUSPERTASK%", batch_pars[2])  #number of tasks per node
batch_lines = batch_lines.replace(r"%MEMPERCPU%", batch_pars[3])  #allocated RAM per cpu node
batch_lines = batch_lines.replace(r"%INPUTFILE%", input_template)  #allocated RAM per cpu node

# Save modified batch file in run folder
new_batch_file = os.path.join(run_folder, batch_file)
with open(new_batch_file, "w") as f:
    f.write(batch_lines)

# Batch Siesta job inside the run folder
command = f"sbatch {batch_file}"
subprocess.run(command, shell=True, executable="/bin/bash", cwd=run_folder)
