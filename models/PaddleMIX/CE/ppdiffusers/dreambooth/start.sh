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
echo "*******dreambooth singe_train begin***********"
(bash single_train.sh) 2>&1 | tee ${log_dir}/dreambooth_singe_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "dreambooth singe_train run success" >>"${log_dir}/ce_res.log"
else
    echo "dreambooth singe_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******dreambooth singe_train end***********"

# 单机训练的结果进行推理
echo "******dreambooth singe infer begin***********"
(python infer.py)2>&1 | tee ${log_dir}/dreambooth_single_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "dreambooth single_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "dreambooth single_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******dreambooth singe infer end***********"

# 多机训练
echo "*******dreambooth muti_train begin***********"
(bash multi_train.sh) 2>&1 | tee ${log_dir}/dreambooth_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "dreambooth multi_train run success" >>"${log_dir}/ce_res.log"
else
    echo "dreambooth multi_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******dreambooth multi_train end***********"

# 多机训练的结果进行推理
echo "*******dreambooth multi infer begin***********"
(python infer.py) 2>&1 | tee ${log_dir}/dreambooth_multi_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "dreambooth multi_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "dreambooth multi_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******dreambooth multi infer end***********"

# 给模型引入先验知识（图片）一同训练
echo "*******dreambooth train_with_class begin***********"
(bash train_with_class.sh) 2>&1 | tee ${log_dir}/dreambooth_train_with_class.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "dreambooth train_with_class run success" >>"${log_dir}/ce_res.log"
else
    echo "dreambooth train_with_class run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******dreambooth train_with_class end***********"

# 给模型引入先验知识（图片）一同训练的结果进行推理
echo "*******dreambooth infer_with_class begin***********"
(python infer_with_class.py) 2>&1 | tee ${log_dir}/dreambooth_infer_with_class.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "dreambooth infer_with_class success" >>"${log_dir}/ce_res.log"
else
    echo "dreambooth infer_with_class fail" >>"${log_dir}/ce_res.log"
fi
echo "*******dreambooth infer_with_class end***********"

# lora train
echo "*******dreambooth lora_train begin***********"
(bash lora_train.sh ) 2>&1 | tee ${log_dir}/dreambooth_lora_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "dreambooth lora_train success" >>"${log_dir}/ce_res.log"
else
    echo "dreambooth lora_train fail" >>"${log_dir}/ce_res.log"
fi
echo "*******dreambooth lora_train end***********"

# lora train
echo "*******dreambooth lora_infer begin***********"
(python lora_infer.py) 2>&1 | tee ${log_dir}/dreambooth_lora_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "dreambooth lora_infer success" >>"${log_dir}/ce_res.log"
else
    echo "dreambooth lora_infer fail" >>"${log_dir}/ce_res.log"
fi
echo "*******dreambooth lora_lora_infer  end***********"

# # 查看结果
# cat ${log_dir}/ce_res.log
rm -rf ${work_path}/dream_outputs/*
rm -rf ${work_path}/dream_outputs_with_class/*
rm -rf ${work_path}/dogs/

echo exit_code:${exit_code}
exit ${exit_code}
