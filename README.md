# DFT

# Monomers
- main_conf: 4 monomers optimised in Avogadro or simple DFT simulation
- conformers_not_optimised: 9 monomers created with RDKit, not optimised

## Procedure to creation of the monomers
I used the package RDKit to create the conformers. The script is in the folder `conformers_creation`. 
- Started with a xyz file of the monomer (2OS-Short_init0.xyz)
- Created a .mol file with Open Babel 
(command: obabel -ixyz 2OS-Short_init0.xyz -omol -O 2OS-Short_init0.mol)
- Used the script `generate_confs.py` to create 9 conformers expressed in a .sdf file
- The .sdf file can be split into individual .sdf, .xyz, .mol files with `split_conformers.py`
- A group of .xyz files can be grouped into a single .sdf file with `xyz2sdf.py`. This is particularly useful for handmade configurations done in Avogadro.
- A collection of .fdf files (input for Siesta simulations) is then generated with the script `generate_fdf_files.py` or `generate_fdf_files_dimers.py`. 
- `generate_fdf_files.py` has two main input arguments:
1. input .sdf file with the atom positions
2. template for the fdf file: contains all the parameters for the DFT simulation including the basis set, pseudopotentials, mesh cutoff, spin initialization etc.

Note that the atom ordering in the main_conf and confoermers_not_optimised is different. Therefore, one should bve careful when initializing the spins in the fdf template. In particular for a monomer in main_conf, the possible radical sites are aroms 3, 18, 29, 42 (with 29 and 42 being the external spins). For a monomer in conformers_not_optimised, the possible radical sites are atoms  23, 7, 29, 42 (with 30 and 43 being the external spins).

# Dimers 
I used the package packmol to create the dimers. The input files are in the folder `dimers_creation`. The output files are in the folder `dimers`.

I used the file 2OS-Short_init0.xyz as the first monomer, and the conformers in the folder `conformers_not_optimised` as the second monomer. The output files are named dimer_1.xyz, dimer_2.xyz, etc.
dimer_0.xyz was created with 2OS-Short_init0.xyz and short-2OS_orthogonal.xyz.

Every dimer is generated running the command 
packmol < dimer.inp
the input file `dimer.inp` is the same for all dimers, except for the output file name and the second monomer structure.

"""
tolerance 2.0          # Å; minimum inter-atomic distance
filetype xyz           # use XYZ in/out
output  dimer_9.xyz      # name of the packed structure
seed -1                # -1 → take the clock, gives a new random pose each run

structure 2OS-Short_init0.xyz
  number 1
  fixed 0. 0. 0.  0. 0. 0.   # keep monomer A at the origin
end structure

structure ../conformers_not_optimised/conf_9.xyz
  number 1
  inside sphere 0. 0. 0. 11  # place B somewhere in a 10 Å sphere around A
end structure
"""

# DelftBlue
For a crush course on using DelftBlue refer to `https://doc.dhpc.tudelft.nl/delftblue/crash-course/`
The main steps are:
- login to DelftBlue with your account. Once setted up it's enough to run `ssh delftblue` from your terminal
- Go to the right folder in the remote server cd /scratch/<username>/...
(Most of the time it will be in `/scratch/rconte1/Short-2OS/relaxation/fdf_files`)
- print the current path with `pwd`
- Open another terminal in your local machine and go to the folder where the files are stored. Then copy the files to DelftBlue with `scp -r <mylocalfolder> <username>@login.delftblue.tudelft.nl:<delftblue_path>

Inside DelftBlue there are the following scripts/folders/files:
- fdf_files: contains all the input files for the DFT simulations
- pseudopotentials: contains the pseudopotentials for the elements in the system
- runs: contains the result of the simulations

- siesta5_relaxation.sh: script to submit a relaxation job to the queue. Many fields are gonna be replaced with the actual values when submitting the job with the python script.
"""
#!/bin/sh
#
#SBATCH --job-name=%EXECUTABLE%
#SBATCH --output=%OUTPUTFILE%
#SBATCH --error=%ERRORFILE%
#SBATCH --partition=compute-p1
#SBATCH --time=%WALLTIME%
#SBATCH --ntasks=%NTASKS%
#SBATCH --cpus-per-task=%CPUSPERTASK%
#SBATCH --mem-per-cpu=%MEMPERCPU%
#SBATCH --account=research-as-qn

module load 2024r1 openmpi
module use /projects/unsupported/spack2024/share/spack/lmod/linux-rhel8-x86_64/Core/
module use /projects/unsupported/spack2024/share/spack/lmod/linux-rhel8-x86_64/openmpi/4.1.6-w6w5qi5/Core/
module use /projects/unsupported/spack2024/share/spack/lmod/linux-rhel8-x86_64/openmpi/4.1.6-h2uag4k/Core/

module load siesta

srun siesta < %INPUTFILE%
"""
- single_run.py: python script to submit a single job to the queue. It replaces the fields in siesta5_relaxation.sh with the actual values and then runs `sbatch siesta5_relaxation.sh` to submit the job.
- folder_run.py: python script to submit multiple jobs to the queue. It loops over all the fdf files in the fdf_files folder and for each of them it creates a folder in the runs folder, copies the fdf file and the pseudopotentials there, and then submits the job with single_run.py. 

Note: if you want to run a job using the final structure of the previous job as input, you need to copy the run result folder of the previous job with a new name, and run the new job in that same location without replacing teh existing files. This way the new job will use the final structure of the previous job as input. 
if you only need to start the new simulation with the old final .xyz structure, you can just copy the final xyz file in the fdf_files folder and use it as input for a new fdf file.


