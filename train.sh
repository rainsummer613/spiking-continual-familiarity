#!/bin/bash

# Training genetic algorithm to find optimal STDP parameters

#SBATCH -J train
#SBATCH --array=0-179 # how many tasks in the array
#SBATCH -t 47:00:00
#SBATCH --cpus-per-task=5    
#SBATCH -o out/train-%a.out
#SBATCH --mem-per-cpu=700mb 

# Load software
# module load Spack
spack load miniconda3 # module load anaconda3
source activate ...  # activate Python environment

# Run python script with a command line argument
srun .../python train.py -c $SLURM_ARRAY_TASK_ID -lo 1
