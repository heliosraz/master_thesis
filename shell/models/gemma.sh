vllm serve google/gemma-3-4b-it --max_num_batched_tokens 7376 \
                        --max_model_len 7376 \
                        --enforce-eager \
                        --structured-outputs-config.backend guidance \
                        --dtype float32