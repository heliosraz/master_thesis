vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B --max_num_batched_tokens 7376 \
                        --max_model_len 7376 \
                        --enforce-eager \
                        --structured-outputs-config.backend guidance \
                        --dtype float32 \
                        --port 8000