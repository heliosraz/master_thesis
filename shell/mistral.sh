#!/bin/bash

#SBATCH --job-name=mistral
#SBATCH --gres=gpu:2
#SBATCH --output=./output/mistral_output.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=40964
#SBATCH --partition=regular

source ~/miniconda3/bin/activate nlp

if [ -n "$1" ]; then
    PYTHONPATH=~/master_thesis python3 ~/master_thesis/experiments/exp1.py 2 $1
else
    PYTHONPATH=~/master_thesis python3 ~/master_thesis/experiments/exp1.py 2
fi
