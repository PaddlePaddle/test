#!/bin/bash

python deploy/llava/run_static_predict.py --model_name_or_path "paddlemix/llava/llava-v1.5-7b" \
--image_file "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg" \
--first_model_path "llava_static/encode_image/clip"  \
--second_model_path "llava_static/encode_text/llama" \
--fp16 \
--benchmark
