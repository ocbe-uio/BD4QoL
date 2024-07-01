#!/bin/bash

# Job name:
#SBATCH --job-name=M8_lasso_conformal
#
# Project:
#SBATCH --account=p1402
#
# Wall time limit:
#SBATCH --time=00-06:00:00
#
#SBATCH --ntasks=20 --cpus-per-task=1 --ntasks-per-node=20
#SBATCH --mem-per-cpu=5G

module --quiet purge  # Reset the modules to the system default
module load singularity/3.7.3
module list

srun -n20 -l --multi-prog jobs/M8_commands.conf
