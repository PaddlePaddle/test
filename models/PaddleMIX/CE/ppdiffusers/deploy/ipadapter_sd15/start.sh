#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}
log_dir=${root_path}/deploy_log
if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi


work_path1=${root_path}/PaddleMIX/ppdiffusers/deploy/ipadapter/sd15
echo ${work_path1}
cd ${cur_path}
/bin/cp -rf ./* ${work_path1}/
exit_code=0

cd ${work_path1}

bash sd15_export.sh
exit_code=$(($exit_code + $?))

bash sd15_infer.sh
exit_code=$(($exit_code + $?))

