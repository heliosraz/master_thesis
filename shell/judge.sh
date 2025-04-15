#!/bin/bash

#SBATCH --job-name=judge
#SBATCH --output=./shell/output/judge_output.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=regular

source ~/miniconda3/bin/activate thesis
python ~/master_thesis/experiments/judges.py