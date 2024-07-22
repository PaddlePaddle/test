#!/bin/bash

python deploy/llava/export_model.py \
    --model_name_or_path "paddlemix/llava/llava-v1.5-7b" \
    --save_path "./llava_static" \
    --encode_image \
    --fp16