#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=./PaddleNLP/:../PaddleMIX

python deploy/qwen_vl/run_static_predict.py \
    --first_model_path "/path/to/checkpoints/encode_image/vision" \
    --second_model_path "/path/to/checkpoints/encode_text/qwen" \
    --model_name_or_path "qwen-vl/qwen-vl-7b-static" \