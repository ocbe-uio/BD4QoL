#!/bin/bash

# Job name:
#SBATCH --job-name=M6_complete
#
# Project:
#SBATCH --account=p1402
#
# Wall time limit:
#SBATCH --time=0-05:00:00
#
#SBATCH --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=5G

module --quiet purge  # Reset the modules to the system default
#module load Python/3.8.2-GCCcore-9.3.0
module load singularity/3.7.3
module list

srun singularity run fermi.sif python3.10 src/M6_logreg_decline.py
