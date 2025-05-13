#!/bin/bash

#SBATCH --job-name=mass_exp1
#SBATCH --partition=regular

sbatch --job-name=llama_1 --output=./shell/output/llama_1_output.txt shell/llama.sh 1
sbatch --job-name=gemma_1 --output=./shell/output/gemma_1_output.txt shell/gemma.sh 1
sbatch --job-name=mist_1 --output=./shell/output/mist_1_output.txt shell/mistral.sh 1
sbatch --job-name=dpseek_1 --output=./shell/output/dpseek_1_output.txt shell/deepseek.sh 1

sbatch --job-name=llama_2 --output=./shell/output/llama_2_output.txt shell/llama.sh 2
sbatch --job-name=gemma_2 --output=./shell/output/gemma_2_output.txt shell/gemma.sh 2
sbatch --job-name=mist_2 --output=./shell/output/mist_2_output.txt shell/mistral.sh 2
sbatch --job-name=dpseek_2 --output=./shell/output/dpseek_2_output.txt shell/deepseek.sh 2

sbatch --job-name=llama_3 --output=./shell/output/llama_3_output.txt shell/llama.sh 3
sbatch --job-name=gemma_3 --output=./shell/output/gemma_3_output.txt shell/gemma.sh 3
sbatch --job-name=mist_3 --output=./shell/output/mist_3_output.txt shell/mistral.sh 3
sbatch --job-name=dpseek_3 --output=./shell/output/dpseek_3_output.txt shell/deepseek.sh 3

sbatch --job-name=llama_4 --output=./shell/output/llama_4_output.txt shell/llama.sh 4
sbatch --job-name=gemma_4 --output=./shell/output/gemma_4_output.txt shell/gemma.sh 4
sbatch --job-name=mist_4 --output=./shell/output/mist_4_output.txt shell/mistral.sh 4
sbatch --job-name=dpseek_4 --output=./shell/output/dpseek_4_output.txt shell/deepseek.sh 4