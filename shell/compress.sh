#!/bin/bash

#SBATCH --job-name=compress
#SBATCH --output=./shell/output/compress_output.txt
#SBATCH --cpus-per-task=4 
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=regular

cd ~/master_thesis
tar -czvf embed_results.tar.gz results/embed/