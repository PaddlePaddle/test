#!/bin/bash


rm -rf temp/checkpoints_short/

export FLAGS_conv_workspace_size_limit=4096
python -u train_lvdm_short.py \
    --do_train \
    --do_eval \
    --label_names pixel_values \
    --eval_steps 20 \
    --vae_type 3d \
    --output_dir temp/checkpoints_short \
    --unet_config_file unet_configs/lvdm_short_sky_no_ema/unet/config.json \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 6e-5 \
    --max_steps 20 \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --image_logging_steps 10 \
    --logging_steps 1 \
    --save_steps 20 \
    --seed 23 \
    --dataloader_num_workers 0 \
    --weight_decay 0.01 \
    --max_grad_norm 0 \
    --overwrite_output_dir False \
    --pretrained_model_name_or_path westfish/lvdm_short_sky_no_ema \
    --train_data_root ./sky_timelapse_lvdm \
    --eval_data_root ./sky_timelapse_lvdm


# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi
