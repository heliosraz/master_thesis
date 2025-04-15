#!/bin/bash

#SBATCH --job-name=mass_eval
#SBATCH --partition=regular

sbatch --job-name=eval1_llama --output=./shell/output/eval_1_llama_output.txt shell/eval.sh 0 1
sbatch --job-name=eval2_llama --output=./shell/output/eval_2_llama_output.txt shell/eval.sh 0 2
sbatch --job-name=eval3_llama --output=./shell/output/eval_3_llama_output.txt shell/eval.sh 0 3
sbatch --job-name=eval4_llama --output=./shell/output/eval_4_llama_output.txt shell/eval.sh 0 4

sbatch --job-name=eval1_gemma --output=./shell/output/eval_1_gemma_output.txt shell/eval.sh 1 1
sbatch --job-name=eval2_gemma --output=./shell/output/eval_2_gemma_output.txt shell/eval.sh 1 2
sbatch --job-name=eval3_gemma --output=./shell/output/eval_3_gemma_output.txt shell/eval.sh 1 3
sbatch --job-name=eval4_gemma --output=./shell/output/eval_4_gemma_output.txt shell/eval.sh 1 4

sbatch --job-name=eval1_mistral --output=./shell/output/eval_1_mistral_output.txt shell/eval.sh 2 1
sbatch --job-name=eval2_mistral --output=./shell/output/eval_2_mistral_output.txt shell/eval.sh 2 2
sbatch --job-name=eval3_mistral --output=./shell/output/eval_3_mistral_output.txt shell/eval.sh 2 3
sbatch --job-name=eval4_mistral --output=./shell/output/eval_4_mistral_output.txt shell/eval.sh 2 4

sbatch --job-name=eval1_dpseek --output=./shell/output/eval_1_dpseek_output.txt shell/eval.sh 3 1
sbatch --job-name=eval2_dpseek --output=./shell/output/eval_2_dpseek_output.txt shell/eval.sh 3 2
sbatch --job-name=eval3_dpseek --output=./shell/output/eval_3_dpseek_output.txt shell/eval.sh 3 3
sbatch --job-name=eval4_dpseek --output=./shell/output/eval_4_dpseek_output.txt shell/eval.sh 3 4
