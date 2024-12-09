#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/paddlemix/examples/YOLO-World/
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

# infer
cd ${work_path}
export FLAGS_use_cuda_managed_memory=true
export FLAGS_allocator_strategy=auto_growth
curl -O https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg
# wget https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505
echo "*******paddlemix yolo-world infer***********"
(python infer.py \
    --config ./configs/yolo_world_x.yml \
    -o weights=./pretrain/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain-8698fbfa.pdparams \
    --image=./000000004505.jpg \
    --text bus\
    --topk=3 \
    --threshold=0.6 \
    --output_dir=./yolo_output) 2>&1 | tee ${log_dir}/paddlemix_yolo-world_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix yolo-world infer run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix yolo-world infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix yolo-world infer end***********"

unset FLAGS_use_cuda_managed_memory
unset FLAGS_allocator_strategy

echo exit_code:${exit_code}
exit ${exit_code}