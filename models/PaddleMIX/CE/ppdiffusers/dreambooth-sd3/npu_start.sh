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
/bin/cp -rf ../change_paddlenlp_version.sh ${work_path}

cd ${work_path}
exit_code=0

bash prepare.sh

# 设置NPU环境变量
export FLAGS_npu_storage_format=0
export FLAGS_use_stride_kernel=0
export FLAGS_npu_scale_aclnn=True
export FLAGS_allocator_strategy=auto_growth
export USE_PEFT_BACKEND=True

# # 单机训练
# echo "*******dreambooth train begin***********"
# (bash train.sh) 2>&1 | tee ${log_dir}/dreambooth_train.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "dreambooth train run success" >>"${log_dir}/ce_res.log"
# else
#     echo "dreambooth train run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******dreambooth train end***********"



# # 单机推理
# echo "*******dreambooth infer begin***********"
# (python infer.py) 2>&1 | tee ${log_dir}/dreambooth_infer.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "dreambooth infer run success" >>"${log_dir}/ce_res.log"
# else
#     echo "dreambooth infer run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******dreambooth infer end***********"

# Lora训练
echo "*******dreambooth lora train begin***********"
(bash lora_train.sh) 2>&1 | tee ${log_dir}/dreambooth_lora_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "dreambooth lora train run success" >>"${log_dir}/ce_res.log"
else
    echo "dreambooth lora train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******dreambooth lora train end***********"

# Lora推理
echo "*******dreambooth lora infer begin***********"
(python lora_infer.py) 2>&1 | tee ${log_dir}/dreambooth_lora_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "dreambooth lora infer run success" >>"${log_dir}/ce_res.log"
else
    echo "dreambooth lora infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******dreambooth lora infer end***********"

# # 查看结果
# cat ${log_dir}/ce_res.log
rm -rf ${work_path}/dogs/

echo exit_code:${exit_code}
exit ${exit_code}
