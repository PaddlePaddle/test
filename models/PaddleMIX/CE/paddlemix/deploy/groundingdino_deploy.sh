#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/deploy/groundingdino/
echo ${work_path}

log_dir=${root_path}/log_deploy

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
exit_code=0

cd ${root_path}/PaddleMIX/paddlemix/models/groundingdino/csrc/
export http_proxy=${proxy}
export https_proxy=${proxy}
python setup_ms_deformable_attn_op.py install
unset http_proxy
unset https_proxy

cd ${work_path}

echo "*******paddlemix deploy groundingdino begin***********"

#静态图模型导出
(python export.py \
    --dino_type GroundingDino/groundingdino-swint-ogc) 2>&1 | tee ${log_dir}/run_deploy_groundingdino_export.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix deploy groundingdino export run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix deploy groundingdino export run fail" >>"${log_dir}/ce_res.log"
fi

#静态图预测
(python predict.py \
    --text_encoder_type GroundingDino/groundingdino-swint-ogc \
    --model_path output_groundingdino/GroundingDino/groundingdino-swint-ogc \
    --input_image https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg \
    --output_dir ./groundingdino_predict_output \
    --prompt "bus") 2>&1 | tee ${log_dir}/run_deploy_groundingdino_predict.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix deploy groundingdino predict run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix deploy groundingdino predict run fail" >>"${log_dir}/ce_res.log"
fi

echo "*******paddlemix deploy groundingdino end***********"

# 检查命令是否成功执行
if [ ${exit_code} -ne 0 ]; then
    exit 1
fi
