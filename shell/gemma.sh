#!/bin/bash

#SBATCH --job-name=gemma_test
#SBATCH --error=gemma_error.txt
#SBATCH --cpus-per-task=1
# SBATCH --mem-per-cpu=40964
# SBATCH --partition=regular

echo "Starting the script..."
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"

source ~/miniconda3/bin/activate nlp
if [ -n "$1" ]; then
    PYTHONPATH=~/master_thesis python3 ~/master_thesis/experiments/exp1.py 1 $1
else
    PYTHONPATH=~/master_thesis python3 ~/master_thesis/experiments/exp1.py 1
fi

