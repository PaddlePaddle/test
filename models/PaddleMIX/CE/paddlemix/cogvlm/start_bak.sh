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
exit_code=0

cd ${work_path}

# 下载依赖、数据集和权重
bash prepare.sh

# 训练
# export FLAGS_use_cuda_managed_memory=true
# export FLAGS_allocator_strategy=auto_growth
echo "*******paddlemix cogvlm infer***********"
(python paddlemix/examples/cogvlm/chat_demo.py \
    --model_name_or_path "THUDM/cogvlm-chat") 2>&1 | tee ${log_dir}/paddlemix_cogvlm_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix cogvlm infer run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix cogvlm infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix cogvlm infer end***********"


echo exit_code:${exit_code}
exit ${exit_code}
