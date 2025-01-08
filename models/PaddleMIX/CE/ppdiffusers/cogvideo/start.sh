#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/cogvideo/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0

#bash prepare.sh




echo "*******CogVideo infer begin***********"
(python infer.py \
  --prompt "a bear is walking in a zoon" \
  --model_path THUDM/CogVideoX-2b \
  --generate_type "t2v" \
  --dtype "float16" \
  --seed 42) 2>&1 | tee ${log_dir}/CogVideo_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "CogVideo_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "CogVideo_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******CogVideo infer end***********"

# # 查看结果
# cat ${log_dir}/ce_res.log
rm -rf ${work_path}/ubcNbili_data/*

echo exit_code:${exit_code}
exit ${exit_code}
