#!/bin/bash
#SBATCH --job-name=gemma_test
#SBATCH --output=test_output.txt
#SBATCH --error=test_error.txt
#SBATCH --cpus-per-task=1

echo "Starting the script..."
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
sleep 20

source ~/miniconda3/bin/activate nlp
echo "Activated conda environment 'nlp'."
# if [ -n "$1" ]; then
#     PYTHONPATH=~/master_thesis python3 ~/master_thesis/experiments/exp1.py 1 $1
# else
#     PYTHONPATH=~/master_thesis python3 ~/master_thesis/experiments/exp1.py 1
# fi

