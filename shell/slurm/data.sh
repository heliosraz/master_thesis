#!/bin/bash

#SBATCH --job-name=data
#SBATCH --output=/output/data_output.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=regular

source ~/miniconda3/bin/activate thesis
PYTHONPATH=~/master_thesis python3 ~/master_thesis/processing/generate_task.py