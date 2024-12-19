#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/
echo ${work_path}

log_dir=${root_path}/paddlemix_examples_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
/bin/cp -f ../check_loss.py ${work_path}/
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
# echo "*******paddlemix internlm_xcomposer2 single_infer***********"
# (python paddlemix/examples/internlm_xcomposer2/chat_demo.py \
#     --model_name_or_path "internlm/internlm-xcomposer2-7b" \
#     --image_path "./000000004505.jpg" \
#     --text "Please describe this image in detail.") 2>&1 | tee ${log_dir}/paddlemix_internlm_xcomposer2_single_infer.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "paddlemix internlm_xcomposer2 single_infer run success" >>"${log_dir}/ce_res.log"
# else
#     echo "paddlemix internlm_xcomposer2 single_infer run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******paddlemix internlm_xcomposer2 single_infer end***********"

export FLAGS_use_cuda_managed_memory=true
export FLAGS_allocator_strategy=auto_growth
export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1

echo "*******paddlemix internlm_xcomposer2 train fp16***********"
(python  -u check_loss.py "python paddlemix/tools/supervised_finetune.py interlm_xcomposer2_sft_argument.json") 2>&1 | tee ${log_dir}/paddlemix_internlm_xcomposer2_train_fp32.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix internlm_xcomposer2 train fp16 run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix internlm_xcomposer2 train fp16 run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix internlm_xcomposer2 train fp16 end***********"

echo exit_code:${exit_code}
exit ${exit_code}
