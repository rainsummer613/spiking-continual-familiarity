#!/bin/bash

# Testing with optimal Hebbian and other parameters for various regimes

#SBATCH -J hebb-test
#SBATCH --array=0-899 # how many tasks in the array
#SBATCH -t 47:00:00
#SBATCH --cpus-per-task=5
#SBATCH -o out/test-%a.out
#SBATCH --mem-per-cpu=700mb

# Load software
# module load Spack
spack load miniconda3 # module load anaconda3
source activate familiarity  # activate Python environment

# Run python script with a command line argument
srun /home/staff/v/vzemliak/.conda/envs/familiarity/bin/python test.py -po 1 -lo 1 -c $SLURM_ARRAY_TASK_ID
