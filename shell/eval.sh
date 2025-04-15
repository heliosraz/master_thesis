#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --output=./shell/output/eval_output.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=regular

source ~/miniconda3/bin/activate thesis
if [ -n "$1" ]; then
    python ~/master_thesis/experiments/eval.py $1
else
    python ~/master_thesis/experiments/eval.py
fi