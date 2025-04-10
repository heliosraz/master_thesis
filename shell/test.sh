#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test_output.txt
#SBATCH --error=test_error.txt
#SBATCH --cpus-per-task=1

echo "Hello, SLURM!"
sleep 10