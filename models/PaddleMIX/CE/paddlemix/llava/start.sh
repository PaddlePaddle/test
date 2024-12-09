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
/bin/cp -f ../check_loss.py ${work_path}/
exit_code=0

cd ${work_path}

# 下载依赖、数据集和权重
bash prepare.sh

# infer
export FLAGS_use_cuda_managed_memory=true
export FLAGS_allocator_strategy=auto_growth

# infer暂时没有达到自动化测试，需要手动测试
# echo "*******paddlemix llava infer***********"
# (python paddlemix/examples/llava/run_predict_multiround.py \
#     --model-path "paddlemix/llava/llava-v1.5-7b" \
#     --image-file "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg") 2>&1 | tee ${log_dir}/paddlemix_llava_infer.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "paddlemix llava infer run success" >>"${log_dir}/ce_res.log"
# else
#     echo "paddlemix llava infer run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******paddlemix llava infer end***********"

echo "*******paddlemix llava finetune***********"
# llava pretain 有报错
(python  -u check_loss.py "python paddlemix/examples/llava/pretrain.py llava_v100_pretrain.json") 2>&1 | tee ${log_dir}/paddlemix_llava_finetune.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix llava finetune run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix llava finetune run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix llava finetune end***********"

echo "*******paddlemix llava sft***********"
(python  -u check_loss.py "python paddlemix/tools/supervised_finetune.py llava_v100_sft.json") 2>&1 | tee ${log_dir}/paddlemix_llava_sft.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix llava sft run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix llava sft run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix llava sft end***********"

echo "*******paddlemix llava lora***********"

(python  -u check_loss.py "python paddlemix/tools/supervised_finetune.py llava_v100_lora.json") 2>&1 | tee ${log_dir}/paddlemix_llava_lora.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix llava lora run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix llava lora run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix llava lora end***********"

unset FLAGS_use_cuda_managed_memory
unset FLAGS_allocator_strategy

echo exit_code:${exit_code}
exit ${exit_code}