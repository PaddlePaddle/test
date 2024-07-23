#!/bin/bash



cd PaddleNLP/llm
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=../../PaddleNLP/:../../../PaddleMIX:../../PaddleNLP/llm
export FLAGS_use_cuda_managed_memory=true
export FLAGS_allocator_strategy=auto_growth
export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
python predict/export_model.py \
    --model_name_or_path "qwen-vl/qwen-vl-7b-static" \
    --output_path ./checkpoints/encode_text/ \
    --dtype float16 \
    --inference_model \
    --model_prefix qwen \
    --model_type qwen-img2txt
unset FLAGS_use_cuda_managed_memory
unset FLAGS_allocator_strategy
unset FLAGS_embedding_deterministic
unset FLAGS_cudnn_deterministic
cd ..
cd ..