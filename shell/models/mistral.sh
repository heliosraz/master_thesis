vllm serve mistralai/Mistral-7B-Instruct-v0.3 --max_num_batched_tokens 7376 \
                        --max_model_len 7376 \
                        --enforce-eager \
                        --structured-outputs-config.backend guidance \
                        --dtype float32\
                        --port 8003