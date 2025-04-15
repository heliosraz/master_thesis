#!/bin/bash

#SBATCH --job-name=mass_eval
#SBATCH --partition=regular

sbatch --job-name=eval_llama --output=./shell/output/eval_llama_output.txt shell/eval.sh 0 1
sbatch --job-name=eval_llama --output=./shell/output/eval_llama_output.txt shell/eval.sh 0 2
sbatch --job-name=eval_llama --output=./shell/output/eval_llama_output.txt shell/eval.sh 0 3
sbatch --job-name=eval_llama --output=./shell/output/eval_llama_output.txt shell/eval.sh 0 4

sbatch --job-name=eval_gemma --output=./shell/output/eval_gemma_output.txt shell/eval.sh 1 1
sbatch --job-name=eval_gemma --output=./shell/output/eval_gemma_output.txt shell/eval.sh 1 2
sbatch --job-name=eval_gemma --output=./shell/output/eval_gemma_output.txt shell/eval.sh 1 3
sbatch --job-name=eval_gemma --output=./shell/output/eval_gemma_output.txt shell/eval.sh 1 4

sbatch --job-name=eval_mistral --output=./shell/output/eval_mistral_output.txt shell/eval.sh 2 1
sbatch --job-name=eval_mistral --output=./shell/output/eval_mistral_output.txt shell/eval.sh 2 2
sbatch --job-name=eval_mistral --output=./shell/output/eval_mistral_output.txt shell/eval.sh 2 3
sbatch --job-name=eval_mistral --output=./shell/output/eval_mistral_output.txt shell/eval.sh 2 4

sbatch --job-name=eval_dpseek --output=./shell/output/eval_dpseek_output.txt shell/eval.sh 3 1
sbatch --job-name=eval_dpseek --output=./shell/output/eval_dpseek_output.txt shell/eval.sh 3 2
sbatch --job-name=eval_dpseek --output=./shell/output/eval_dpseek_output.txt shell/eval.sh 3 3
sbatch --job-name=eval_dpseek --output=./shell/output/eval_dpseek_output.txt shell/eval.sh 3 4
