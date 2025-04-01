#!/bin/bash

#SBATCH --job-name=gemma
#SBATCH --output=./output/gemma_output.txt
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=40964
#SBATCH --partition=regular

source ~/miniconda3/bin/activate nlp
if [ -n "$1" ]; then
    PYTHONPATH=$PWD python3 ./experiments/exp1.py 1 $1
else
    PYTHONPATH=$PWD python3 ./experiments/exp1.py 1
fi

