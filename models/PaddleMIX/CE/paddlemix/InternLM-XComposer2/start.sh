#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/
echo ${work_path}

log_dir=${root_path}/log_mix

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
exit_code=0

cd ${work_path}

# 下载依赖、数据集和权重
(bash prepare.sh) 2>&1 | tee ${log_dir}/prepare_internlm_xcomposer2.log

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=${1:-0}

# 单轮预测
export FLAGS_use_cuda_managed_memory=true
export FLAGS_allocator_strategy=auto_growth
export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
echo "*******paddlemix internlm_xcomposer2 single_infer***********"
(python paddlemix/examples/internlm_xcomposer2/chat_demo.py \
    --model_name_or_path "internlm/internlm-xcomposer2-7b" \
    --image_path "./000000004505.jpg" \
    --text "Please describe this image in detail.") 2>&1 | tee ${log_dir}/paddlemix_internlm_xcomposer2_single_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix internlm_xcomposer2 single_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix internlm_xcomposer2 single_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix internlm_xcomposer2 single_infer end***********"
unset FLAGS_use_cuda_managed_memory
unset FLAGS_allocator_strategy
export FLAGS_use_cuda_managed_memory=true
export FLAGS_allocator_strategy=auto_growth
export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
echo "*******paddlemix internlm_xcomposer2 train fp32***********"
(python paddlemix/tools/supervised_finetune.py interlm_xcomposer2_sft_argument.json) 2>&1 | tee ${log_dir}/paddlemix_internlm_xcomposer2_train_fp32.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix internlm_xcomposer2 train fp32 run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix internlm_xcomposer2 train fp32 run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix internlm_xcomposer2 train fp32 end***********"
echo exit_code:${exit_code}
exit ${exit_code}

export FLAGS_use_cuda_managed_memory=true
export FLAGS_allocator_strategy=auto_growth
export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
echo "*******paddlemix internlm_xcomposer2 train bf16 O2***********"
(python paddlemix/tools/supervised_finetune.py interlm_xcomposer2_sft_argument.json) 2>&1 | tee ${log_dir}/paddlemix_internlm_xcomposer2_train_bf16_O2.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix internlm_xcomposer2 train bf16 O2 run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix internlm_xcomposer2 train bf16 O2 run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix internlm_xcomposer2 train bf16 O2 end***********"
echo exit_code:${exit_code}
exit ${exit_code}
