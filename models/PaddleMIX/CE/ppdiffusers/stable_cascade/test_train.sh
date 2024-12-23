export CUDA_VISIBLE_DEVICES=0
export GLOG_minloglevel=7
USE_AMP=False
fp16_opt_level="O2"
enable_tensorboard=True

TRAINING_PYTHON="python -u"
${TRAINING_PYTHON} train_stage_c_trainer.py \
    --do_train \
    --dataset_path /root/lxl/0_SC/Paddle-SC/dataset/haerbin \
    --output_dir ./train_output \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.0e-4 \
    --resolution 512 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_steps 1000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 1000000 \
    --logging_steps 1 \
    --save_steps 5000 \
    --save_total_limit 50 \
    --seed 1 \
    --dataloader_num_workers 0 \
    --num_inference_steps 200 \
    --model_max_length 77 \
    --fp16 ${USE_AMP} \
    --fp16_opt_level=${fp16_opt_level}