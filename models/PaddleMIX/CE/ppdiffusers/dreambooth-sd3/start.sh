#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/dreambooth/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0

bash prepare.sh



# 单机训练
echo "*******dreambooth_sd3 train begin***********"
(bash gpu_train.sh) 2>&1 | tee ${log_dir}/dreambooth_sd3_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "dreambooth_sd3 train run success" >>"${log_dir}/ce_res.log"
else
    echo "dreambooth_sd3 train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******dreambooth_sd3 train end***********"



# 单机推理
echo "*******dreambooth_sd3 infer begin***********"
(python infer.py) 2>&1 | tee ${log_dir}/dreambooth_sd3_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "dreambooth_sd3 infer run success" >>"${log_dir}/ce_res.log"
else
    echo "dreambooth_sd3 infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******dreambooth_sd3 infer end***********"

# Lora训练
echo "*******dreambooth_sd3 lora train begin***********"
(bash gpu_lora_train.sh) 2>&1 | tee ${log_dir}/dreambooth_sd3_lora_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "dreambooth_sd3 lora train run success" >>"${log_dir}/ce_res.log"
else
    echo "dreambooth_sd3 lora train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******dreambooth_sd3 lora train end***********"

# Lora推理
echo "*******dreambooth_sd3 lora infer begin***********"
(python lora_infer.py) 2>&1 | tee ${log_dir}/dreambooth_sd3_lora_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "dreambooth_sd3 lora infer run success" >>"${log_dir}/ce_res.log"
else
    echo "dreambooth_sd3 lora infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******dreambooth_sd3 lora infer end***********"

# # 查看结果
# cat ${log_dir}/ce_res.log
rm -rf ${work_path}/dogs/

echo exit_code:${exit_code}
exit ${exit_code}
