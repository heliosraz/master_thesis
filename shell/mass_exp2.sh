#!/bin/bash

#SBATCH --job-name=mass_exp2
#SBATCH --partition=regular

sbatch --job-name=exp_2_llama --output=./shell/output/exp_2_llama_output.txt shell/exp2.sh 0
sbatch --job-name=exp_2_gemma --output=./shell/output/exp_2_gemma_output.txt shell/exp2.sh 1
sbatch --job-name=exp_2_mist --output=./shell/output/exp_2_mist_output.txt shell/exp2.sh 2
sbatch --job-name=exp_2_dpseek --output=./shell/output/exp_2_dpseek_output.txt shell/exp2.sh 3