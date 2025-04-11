#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=regular


source ~/miniconda3/bin/activate nlp
if [ -n "$1" ]; then
    python3 ~/master_thesis/experiments/exp1.py 1 $1
else
    python3 ~/master_thesis/experiments/exp1.py 1
fi

