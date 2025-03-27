#!/bin/bash

#SBATCH --job-name=experiments
#SBATCH --output=experiment_output.txt
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=40964
#SBATCH --partition=regular

source ~/miniconda3/bin/activate nlp
PYTHONPATH=$PWD python3 ./experiments/exp1.py
