#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/stable_cascade/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0

bash prepare.sh




echo "*******stable_cascade infer begin***********"
(python inference.py) 2>&1 | tee ${log_dir}/stable_cascade_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "stable_cascade_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "stable_cascade_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******stable_cascade infer end***********"


# echo "*******stable_cascade train begin***********"
# (sh train.sh) 2>&1 | tee ${log_dir}/stable_cascade_train.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "stable_cascade_train run success" >>"${log_dir}/ce_res.log"
# else
#     echo "stable_cascade_train run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******stable_cascade train end***********"



# # 查看结果
# cat ${log_dir}/ce_res.log
rm -rf ${work_path}/ubcNbili_data/*

echo exit_code:${exit_code}
exit ${exit_code}
