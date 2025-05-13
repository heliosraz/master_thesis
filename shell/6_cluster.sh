#!/bin/bash

#SBATCH --job-name=cluster
#SBATCH --output=./shell/output/cluster_output.txt
#SBATCH --cpus-per-task=4 
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=regular

source ~/miniconda3/bin/activate thesis
PYTHONPATH=~/master_thesis python3 ~/master_thesis/experiments/cluster.py