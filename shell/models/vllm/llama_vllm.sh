vllm serve meta-llama/Llama-3.2-3B-Instruct  --max_num_batched_tokens 32768 \
                        --max_model_len 32768 \
                        --max-num-seqs 256 \
                        --enforce-eager \
                        --dtype float32\
                        --port 8002