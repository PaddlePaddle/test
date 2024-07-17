#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/deploy/sam/
echo ${work_path}

log_dir=${root_path}/log_deploy

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
exit_code=0

cd ${work_path}

echo "*******paddlemix deploy sam begin***********"

#导出输入类型是 bbox 的静态图
(python export.py \
    --model_type Sam/SamVitH-1024 \
    --input_type boxs \
    --save_dir sam_export) 2>&1 | tee ${log_dir}/run_deploy_sam_box_export.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix deploy sam box export run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix deploy sam box export run fail" >>"${log_dir}/ce_res.log"
fi

#导出输入类型是 points 的静态图
(python export.py \
    --model_type Sam/SamVitH-1024 \
    --input_type points \
    --save_dir sam_export) 2>&1 | tee ${log_dir}/run_deploy_sam_point_export.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix deploy sam point export run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix deploy sam point export run fail" >>"${log_dir}/ce_res.log"
fi

#bbox 提示词推理
(python predict.py \
    --input_image https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg \
    --box_prompt 112 118 513 382 \
    --input_type boxs \
    --model_name_or_path Sam/SamVitH-1024 \
    --cfg Sam/SamVitH-1024_boxs/deploy.yaml) 2>&1 | tee ${log_dir}/run_deploy_sam_box_predict.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix deploy sam box predict run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix deploy sam box predict run fail" >>"${log_dir}/ce_res.log"
fi

#points 提示词推理
(python predict.py \
    --input_image https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg \
    --points_prompt 548 372 \
    --input_type points \
    --model_name_or_path Sam/SamVitH-1024 \
    --cfg Sam/SamVitH-1024_points/deploy.yaml) 2>&1 | tee ${log_dir}/run_deploy_sam_point_predict.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix deploy sam point predict run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix deploy sam point predict run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix deploy sam end***********"

# 检查命令是否成功执行
if [ ${exit_code} -ne 0 ]; then
    exit 1
fi
