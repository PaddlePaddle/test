export MODEL_NAME="stabilityai/stable-video-diffusion-img2vid-xt"
export DATASET_NAME="bdd100k"
export OUTPUT_DIR="sdv_train_output"
export VALID_DATA="valid_image"
export GLOG_minloglevel=2
export FLAGS_conv_workspace_size_limit=4096

python train_image_to_video_svd.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=1 \
    --max_train_steps=20 \
    --width=512 \
    --height=320 \
    --checkpointing_steps=1000 --checkpoints_total_limit=10 \
    --learning_rate=1e-5 --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=200 \
    --output_dir=$OUTPUT_DIR \
    --train_data_dir=$DATASET_NAME \
    --valid_data_path=$VALID_DATA