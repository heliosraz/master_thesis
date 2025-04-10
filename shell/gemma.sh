#!/bin/bash

#SBATCH --job-name=gemma
#SBATCH --output=./output/gemma_output.txt
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=40964
#SBATCH --partition=regular
#SBATCH --error=./output/gemma_error.txt

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

