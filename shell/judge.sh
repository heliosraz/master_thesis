#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 
#SBATCH --mem-per-cpu=16384

source ~/miniconda3/bin/activate thesis
if [ -n "$1" ] && [ -n "$2" ]; then
    python ~/master_thesis/experiments/judge.py $1 $2
elif [ -n "$1" ]; then
    python ~/master_thesis/experiments/judge.py $1
else
    python ~/master_thesis/experiments/judge.py
fi