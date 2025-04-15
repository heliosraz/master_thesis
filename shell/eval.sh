#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 
#SBATCH --mem-per-cpu=16384

source ~/miniconda3/bin/activate thesis
if [ -n "$1" ] && [ -n "$2" ]; then
    python ~/master_thesis/experiments/eval.py $1 $2
elif [ -n "$1" ]; then
    python ~/master_thesis/experiments/eval.py $1
else
    python ~/master_thesis/experiments/eval.py
fi

cat shell/output/eval_1_llama_output.txt
cat shell/output/eval_1_gemma_output.txt
cat shell/output/eval_1_mistral_output.txt
cat shell/output/eval_1_dpseek_output.txt

cat shell/output/eval_2_llama_output.txt
cat shell/output/eval_2_gemma_output.txt
cat shell/output/eval_2_mistral_output.txt
cat shell/output/eval_2_dpseek_output.txt

cat shell/output/eval_3_llama_output.txt
cat shell/output/eval_3_gemma_output.txt
cat shell/output/eval_3_mistral_output.txt
cat shell/output/eval_3_dpseek_output.txt

cat shell/output/eval_4_llama_output.txt
cat shell/output/eval_4_gemma_output.txt
cat shell/output/eval_4_mistral_output.txt
cat shell/output/eval_4_dpseek_output.txt