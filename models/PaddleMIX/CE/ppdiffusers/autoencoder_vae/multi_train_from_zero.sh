python -u -m paddle.distributed.launch --gpus "2,3" train_vae.py \
    --from_scratch \
    --vae_config_file config/vae.json \
    --input_size 256 256 \
    --max_train_steps 100 \
    --learning_rate 1e-4 \
    --batch_size 4 \
    --num_workers 8 \
    --logging_steps 100 \
    --save_steps 2000 \
    --image_logging_steps 500 \
    --disc_start 50001 \
    --kl_weight 0.000001 \
    --disc_weight 0.5 \
    --resolution 512


# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi