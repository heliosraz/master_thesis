#!/bin/bash

#SBATCH --job-name=llama
#SBATCH --output=./output/llama_output.txt
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=40964
#SBATCH --partition=regular

source ~/miniconda3/bin/activate nlp
if [ -n "$1" ]; then
    PYTHONPATH=~/master_thesis python3 ~/master_thesis/experiments/exp1.py 0 $1
else
    PYTHONPATH=~/master_thesis python3 ~/master_thesis/experiments/exp1.py 0
fi

