#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/InstantID/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0

bash prepare.sh




echo "*******InstantID infer begin***********"
(python infer.py) 2>&1 | tee ${log_dir}/InstantID_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "InstantID_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "InstantID_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******InstantID infer end***********"

echo "*******InstantID infer begin***********"
(python infer_lora.py) 2>&1 | tee ${log_dir}/InstantID_infer_lora.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "InstantID_infer lora run success" >>"${log_dir}/ce_res.log"
else
    echo "InstantID_infer lora run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******InstantID infer lora end***********"

# # 查看结果
# cat ${log_dir}/ce_res.log
rm -rf ${work_path}/ubcNbili_data/*

echo exit_code:${exit_code}
exit ${exit_code}
