#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}
log_dir=${root_path}/deploy_log
if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi
export PYTHONPATH=$PYTHONPATH:${cur_path}/PaddleMIX


work_path2=${root_path}/PaddleMIX/ppdiffusers/deploy/sd3
echo ${work_path2}

/bin/cp -rf ./* ${work_path2}/

cd ${work_path2}

# 安装 triton
python -m pip install triton
python -m pip install git+https://github.com/zhoutianzi666/UseTritonInPaddle.git
python -c "import use_triton_in_paddle; use_triton_in_paddle.make_triton_compatible_with_paddle()"

#  测试注意事项 V100 上不支持 cutlass
# 在A100上编译算子也很难·
sed -i 's/exp_enable_use_cutlass=True/exp_enable_use_cutlass=False/g' ./text_to_image_generation-stable_diffusion_3.py


export FLAGS_use_cuda_managed_memory=true
export USE_PPXFORMERS=False
export FLAGS_allocator_strategy=auto_growth
export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1

# infernece
(python  text_to_image_generation-stable_diffusion_3.py \
    --dtype float16 \
    --height 512 \
    --width 512 \
    --num-inference-steps 50 --inference_optimize 1  \
    --benchmark 1) 2>&1 | tee ${log_dir}/sd3_inference.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sd3 inference success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sd3 inference fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sd3 inference end***********"

# 多卡推理
(python -m paddle.distributed.launch --gpus 0,1 text_to_image_generation-stable_diffusion_3.py \
    --dtype float16 \
    --height 512 --width 512 \
    --num-inference-steps 50 \
    --inference_optimize 1 \
    --inference_optimize_bp 1 \
    --benchmark 1) 2>&1 | tee ${log_dir}/sd3_inference_multi.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sd3 inference_multi success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sd3 inference_multi fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sd3 inference_multi end***********"



echo exit_code:${exit_code}
exit ${exit_code}