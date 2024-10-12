export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./dogs"
export OUTPUT_DIR="./lora_dream_outputs"

python train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="visualdl" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=50 \
  --lora_rank=4 \
  --seed=0

# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi
