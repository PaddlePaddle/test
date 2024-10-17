export FLAGS_npu_storage_format=0
export FLAGS_use_stride_kernel=0
export FLAGS_npu_scale_aclnn=True
export FLAGS_allocator_strategy=auto_growth


export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/naruto-blip-captions"

export HF_ENDPOINT=https://hf-mirror.com
export FLAGS_conv_workspace_size_limit=4096

# wandb offline

python -u train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=1024 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=2 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --max_train_steps=30 \
  --checkpointing_steps=10 \
  --output_dir="sd-pokemon-model-lora-sdxl" \
  --validation_prompt="cute dragon creature" --report_to="wandb"

