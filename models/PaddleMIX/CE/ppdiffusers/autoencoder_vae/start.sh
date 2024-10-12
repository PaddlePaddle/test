#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/autoencoder/vae/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0

# 下载依赖和数据
bash prepare.sh

# 单机训练
echo "*******autoencoder_vae singe_train begin***********"
(bash single_train.sh) 2>&1 | tee ${log_dir}/autoencoder_vae_singe_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "autoencoder_vae singe_train run success" >>"${log_dir}/ce_res.log"
else
    echo "autoencoder_vae singe_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******autoencoder_vae singe_train end***********"

# 单机训练的结果进行推理
echo "******autoencoder_vae singe infer begin***********"
(python infer.py 2>&1) | tee ${log_dir}/autoencoder_vae_single_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "autoencoder_vae single_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "autoencoder_vae single_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******autoencoder_vae singe infer end***********"

# 多机训练
echo "*******autoencoder_vae muti_train begin***********"
(bash multi_train.sh) 2>&1 | tee ${log_dir}/autoencoder_vae_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "autoencoder_vae multi_train run success" >>"${log_dir}/ce_res.log"
else
    echo "autoencoder_vae multi_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******autoencoder_vae multi_train end***********"

# 多机训练的结果进行推理
echo "*******autoencoder_vae multi infer begin***********"
(python infer.py) 2>&1 | tee ${log_dir}/autoencoder_vae_multi_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "autoencoder_vae multi_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "autoencoder_vae multi_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******autoencoder_vae multi infer end***********"

#Encoder和Decoder从零开启训练
echo "*******autoencoder_vae singe_train_from_zero begin***********"
(bash single_train_from_zero.sh) 2>&1 | tee ${log_dir}/autoencoder_vae_single_train_zero.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "autoencoder_vae single_train_zero run success" >>"${log_dir}/ce_res.log"
else
    echo "autoencoder_vae single_train_zero run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******autoencoder_vae single_train_zero end***********"

#Encoder和Decoder从零开启训练
echo "*******autoencoder_vae multi_train_from_zero begin***********"
(bash multi_train_from_zero.sh) 2>&1 | tee ${log_dir}/autoencoder_vae_multi_train_zero.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "autoencoder_vae multi_train_zero run success" >>"${log_dir}/ce_res.log"
else
    echo "autoencoder_vae multi_train_zero run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******autoencoder_vae multi_train_zero end***********"

# # 查看结果
# cat ${log_dir}/ce_res.log
rm -rf ${work_path}/autoencoder_outputs/*
rm -rf ${work_path}/data/

echo exit_code:${exit_code}
exit ${exit_code}
