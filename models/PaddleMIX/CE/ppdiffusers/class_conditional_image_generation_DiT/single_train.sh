TRAINING_MODEL_RESUME="None"
TRAINER_INSTANCES='127.0.0.1'
MASTER='127.0.0.1:8080'
TRAINERS_NUM=1 # nnodes, machine num
TRAINING_GPUS_PER_NODE=8 # nproc_per_node
DP_DEGREE=1 # dp_parallel_degree
MP_DEGREE=1 # tensor_parallel_degree
SHARDING_DEGREE=8 # sharding_parallel_degree

# real dp_parallel_degree = nnodes * nproc_per_node / tensor_parallel_degree / sharding_parallel_degree
# Please make sure: nnodes * nproc_per_node >= tensor_parallel_degree * sharding_parallel_degree

config_file=config/DiT_XL_patch2.json
OUTPUT_DIR=./output_trainer/DiT_XL_patch2_trainer
feature_path=./fastdit_imagenet256_tiny

per_device_train_batch_size=32
gradient_accumulation_steps=1

num_workers=4
max_steps=50
logging_steps=10
save_steps=50
image_logging_steps=-1
seed=0

max_grad_norm=-1

USE_AMP=True
FP16_OPT_LEVEL="O1"

enable_tensorboard=True
recompute=True
enable_xformers=True

transformer_engine_backend=False
use_fp8=False # This option takes effect only when transformer_engine_backend=True

TRAINING_PYTHON="python -m paddle.distributed.launch --master ${MASTER} --nnodes ${TRAINERS_NUM} --nproc_per_node ${TRAINING_GPUS_PER_NODE} --ips ${TRAINER_INSTANCES}"
${TRAINING_PYTHON} train_image_generation_trainer.py \
    --do_train \
    --feature_path ${feature_path} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --max_steps ${max_steps} \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps ${image_logging_steps} \
    --logging_dir ${OUTPUT_DIR}/tb_log \
    --logging_steps ${logging_steps} \
    --save_steps ${save_steps} \
    --save_total_limit 50 \
    --dataloader_num_workers ${num_workers} \
    --vae_name_or_path stabilityai/sd-vae-ft-mse \
    --config_file ${config_file} \
    --num_inference_steps 25 \
    --use_ema True \
    --max_grad_norm ${max_grad_norm} \
    --overwrite_output_dir True \
    --disable_tqdm True \
    --fp16_opt_level ${FP16_OPT_LEVEL} \
    --seed ${seed} \
    --recompute ${recompute} \
    --enable_xformers_memory_efficient_attention ${enable_xformers} \
    --bf16 ${USE_AMP} \
    --dp_degree ${DP_DEGREE} \
    --tensor_parallel_degree ${MP_DEGREE} \
    --sharding_parallel_degree ${SHARDING_DEGREE} \
    --sharding "stage1" \
    --hybrid_parallel_topo_order "sharding_first" \
    --amp_master_grad 1 \
    --pipeline_parallel_degree 1 \
    --sep_parallel_degree 1 \
    --transformer_engine_backend ${transformer_engine_backend} \
    --use_fp8 ${use_fp8}

rm -rf ${OUTPUT_DIR}
