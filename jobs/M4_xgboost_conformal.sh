#!/bin/bash

# Job name:
#SBATCH --job-name=M4_xgb_conf
#
# Project:
#SBATCH --account=p1402
#
# Wall time limit:
#SBATCH --time=0-01:00:00
#
#SBATCH --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=5G

module --quiet purge  # Reset the modules to the system default
module load singularity/3.7.3
module list

srun singularity run fermi.sif python3.10 src/M4_xgboost_conformal.py
