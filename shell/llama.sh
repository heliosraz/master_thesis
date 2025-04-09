#!/bin/bash

#SBATCH --job-name=llama
#SBATCH --output=./output/llama_output.txt
#SBATCH --error=./output/llama_error.txt
#SBATCH --gres=gpu:1
#SBATCH --partition=regular

source ~/miniconda3/bin/activate nlp
echo "TESTING"
if [ -n "$1" ]; then
    python3 ~/master_thesis/experiments/exp1.py 0 $1
else
    python3 ~/master_thesis/experiments/exp1.py 0
fi

