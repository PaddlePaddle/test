#!/bin/bash

export DATASET_NAME="lambdalabs/naruto-blip-captions"

python -u train_text_to_image_prior_lora.py \
    --dataset_name=$DATASET_NAME \
    --resolution=768 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=100 \
    --learning_rate=1e-04 \
    --max_grad_norm=1 \
    --checkpoints_total_limit=3 \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --seed=42 \
    --lora_rank=4 \
    --validation_prompt="cute dragon creature" \
    --output_dir="kandi22-prior-pokemon-lora"
