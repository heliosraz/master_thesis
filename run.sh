#!/bin/bash

#SBATCH --job-name=data
#SBATCH --output=output.txt
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=40964

source ~/miniconda3/bin/activate nlp
python3 data_gen.py
