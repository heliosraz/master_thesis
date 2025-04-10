#!/bin/bash

#SBATCH --output=./shell/output/mistral_output.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=40964
#SBATCH --partition=regular

source ~/miniconda3/bin/activate nlp

if [ -n "$1" ]; then
    PYTHONPATH=~/master_thesis python3 ~/master_thesis/experiments/exp1.py 2 $1
else
    PYTHONPATH=~/master_thesis python3 ~/master_thesis/experiments/exp1.py 2
fi
