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
bash prepare.sh

# 训练
export FLAGS_use_cuda_managed_memory=true
export FLAGS_allocator_strategy=auto_growth
echo "*******paddlemix internlm_xcomposer2 single_infer***********"
(python paddlemix/examples/internlm_xcomposer2/chat_demo.py \
    --model_name_or_path "internlm/internlm-xcomposer2-7b" \
    --image_path "path/to/image.jpg" \
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

echo "*******paddlemix internlm_xcomposer2 sft_A100_80G***********"
(paddlemix/tools/supervised_finetune.py paddlemix/config/internlm_xcomposer2/sft_argument.json) 2>&1 | tee ${log_dir}/paddlemix_internlm_xcomposer2_sft_A100_80G.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix internlm_xcomposer2 sft_A100_80G run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix internlm_xcomposer2 sft_A100_80G run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix internlm_xcomposer2 sft_A100_80G end***********"
echo exit_code:${exit_code}
exit ${exit_code}
