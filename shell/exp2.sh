#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=regular
#SBATCH --job-name=exp_2
#SBATCH --output=./shell/output/exp_2_output.txt


source ~/miniconda3/bin/activate thesis
if [ -n "$1" ] && [ -n "$2" ]; then
    python ~/master_thesis/experiments/exp2.py $1 $2
else
    python ~/master_thesis/experiments/exp2.py
fi
