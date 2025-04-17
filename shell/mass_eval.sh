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

#############

sbatch --job-name=eval1_llama_dpseek --output=./shell/output/eval_1_llama_dpseek_output.txt shell/eval.sh 0 DeepSeek-R1-Distill-Llama-8B-task1.json
sbatch --job-name=eval1_gemma_dpseek --output=./shell/output/eval_1_gemma_dpseek_output.txt shell/eval.sh 1 DeepSeek-R1-Distill-Llama-8B-task1.json
sbatch --job-name=eval1_mistral_dpseek --output=./shell/output/eval_1_mistral_dpseek_output.txt shell/eval.sh 2 DeepSeek-R1-Distill-Llama-8B-task1.json
sbatch --job-name=eval1_dpseek_dpseek --output=./shell/output/eval_1_dpseek_dpseek_output.txt shell/eval.sh 3 DeepSeek-R1-Distill-Llama-8B-task1.json

sbatch --job-name=eval1_llama_12 --output=./shell/output/eval_1_llama_gemma12_output.txt shell/eval.sh 0 gemma-3-12b-it-task1.json
sbatch --job-name=eval1_gemma_12 --output=./shell/output/eval_1_gemma_gemma12_output.txt shell/eval.sh 1 gemma-3-12b-it-task1.json
sbatch --job-name=eval1_mistral_12 --output=./shell/output/eval_1_mistral_gemma12_output.txt shell/eval.sh 2 gemma-3-12b-it-task1.json
sbatch --job-name=eval1_dpseek_12 --output=./shell/output/eval_1_dpseek_gemma12_output.txt shell/eval.sh 3 gemma-3-12b-it-task1.json

sbatch --job-name=eval1_llama_llama --output=./shell/output/eval_1_llama_llama_output.txt shell/eval.sh 0 Llama-3.2-3B-Instruct-task1.json
sbatch --job-name=eval1_gemma_llama --output=./shell/output/eval_1_gemma_llama_output.txt shell/eval.sh 1 Llama-3.2-3B-Instruct-task1.json
sbatch --job-name=eval1_mistral_llama --output=./shell/output/eval_1_mistral_llama_output.txt shell/eval.sh 2 Llama-3.2-3B-Instruct-task1.json
sbatch --job-name=eval1_dpseek_llama --output=./shell/output/eval_1_dpseek_llama_output.txt shell/eval.sh 3 Llama-3.2-3B-Instruct-task1.json

sbatch --job-name=eval1_llama_mist --output=./shell/output/eval_1_llama_mist_output.txt shell/eval.sh 0 Mistral-7B-Instruct-v0.3-task1.json
sbatch --job-name=eval1_gemma_mist --output=./shell/output/eval_1_gemma_mist_output.txt shell/eval.sh 1 Mistral-7B-Instruct-v0.3-task1.json
sbatch --job-name=eval1_mistral_mist --output=./shell/output/eval_1_mistral_mist_output.txt shell/eval.sh 2 Mistral-7B-Instruct-v0.3-task1.json
sbatch --job-name=eval1_dpseek_mist --output=./shell/output/eval_1_dpseek_mist_output.txt shell/eval.sh 3 Mistral-7B-Instruct-v0.3-task1.json