#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=./PaddleNLP/:../PaddleMIX:./PaddleNLP/llm

cd PaddleNLP/llm

python predict/export_model.py \
    --model_name_or_path "qwen-vl/qwen-vl-7b-static" \
    --output_path ./checkpoints/encode_text/ \
    --dtype float16 \
    --inference_model \
    --model_prefix qwen \
    --model_type qwen-img2txt

cd ..
cd ..