#!/bin/bash

#SBATCH --job-name=data
#SBATCH --output=data_output.txt
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=40964
#SBATCH --partition=regular

source ~/miniconda3/bin/activate nlp
PYTHONPATH=$PWD python3 ./processing/generate_task.py