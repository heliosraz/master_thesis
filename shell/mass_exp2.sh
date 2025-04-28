#!/bin/bash

#SBATCH --job-name=mass_exp2
#SBATCH --partition=regular

sbatch --job-name=exp_2_llama_gen --output=./shell/output/exp_2_llama_gen_output.txt shell/exp2.sh 0 general
sbatch --job-name=exp_2_gemma_gen --output=./shell/output/exp_2_gemma_gen_output.txt shell/exp2.sh 1 general
sbatch --job-name=exp_2_mist_gen --output=./shell/output/exp_2_mist_gen_output.txt shell/exp2.sh 2 general
sbatch --job-name=exp_2_dpseek_gen --output=./shell/output/exp_2_dpseek_gen_output.txt shell/exp2.sh 3 general

sbatch --job-name=exp_2_llama_res --output=./shell/output/exp_2_llama_res_output.txt shell/exp2.sh 0 results
sbatch --job-name=exp_2_gemma_res --partition=gpu48g --output=./shell/output/exp_2_gemma_res_output.txt shell/exp2.sh 1 results
sbatch --job-name=exp_2_mist_res --partition=gpu48g --output=./shell/output/exp_2_mist_res_output.txt shell/exp2.sh 2 results
sbatch --job-name=exp_2_dpseek_res --output=./shell/output/exp_2_dpseek_res_output.txt --partition=gpu48g shell/exp2.sh 3 results