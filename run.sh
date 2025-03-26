#!/bin/bash

#SBATCH --job-name=task1_thes
#SBATCH --output=output.txt
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=40964

source ~/miniconda3/bin/activate nlp
python3 ./experiments/exp1.py 0 1
