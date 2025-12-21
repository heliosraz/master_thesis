vllm serve meta-llama/Llama-3.2-3B-Instruct --max_num_batched_tokens 7376 \
                        --max_model_len 7376 \
                        --enforce-eager \
                        --structured-outputs-config.backend guidance \
                        --dtype float32