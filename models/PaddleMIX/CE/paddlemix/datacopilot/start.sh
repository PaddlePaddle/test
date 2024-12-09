#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/paddlemix/examples/datacopilot/
echo ${work_path}

log_dir=${root_path}/paddlemix_examples_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
exit_code=0

cd ${work_path}

# 下载依赖、数据集和权重

# importoutput
cd ${work_path}
export FLAGS_use_cuda_managed_memory=true
export FLAGS_allocator_strategy=auto_growth

echo "*******paddlemix datacopilot importoutput***********"
(python importoutput.py) 2>&1 | tee ${log_dir}/paddlemix_datacopilot_importoutput.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix datacopilot importoutput run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix datacopilot importoutput run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix datacopilot importoutput end***********"

unset FLAGS_use_cuda_managed_memory
unset FLAGS_allocator_strategy

echo exit_code:${exit_code}
exit ${exit_code}