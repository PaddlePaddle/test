#!/bin/bash

rm -rf temp/checkpoints_text2video/

export FLAGS_conv_workspace_size_limit=4096
python -u train_lvdm_text2video.py \
    --do_train \
    --do_eval \
    --label_names pixel_values \
    --eval_steps 1000 \
    --vae_type 2d \
    --vae_name_or_path  None \
    --output_dir temp/checkpoints_text2video \
    --unet_config_file unet_configs/lvdm_text2video_orig_webvid_2m/unet/config.json \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 6e-5 \
    --max_steps 20 \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --image_logging_steps 100 \
    --logging_steps 50 \
    --save_steps 20 \
    --seed 23 \
    --dataloader_num_workers 8 \
    --weight_decay 0.01 \
    --max_grad_norm 0 \
    --overwrite_output_dir True \
    --pretrained_model_name_or_path westfish/lvdm_text2video_orig_webvid_2m \
    --recompute True \
    --fp16 \
    --fp16_opt_level O1

# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi
