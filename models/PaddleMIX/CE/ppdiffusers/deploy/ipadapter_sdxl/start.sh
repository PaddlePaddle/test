#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}
log_dir=${root_path}/deploy_log
if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

work_path2=${root_path}/PaddleMIX/ppdiffusers/deploy/ipadapter/sdxl
echo ${work_path2}

/bin/cp -rf ./* ${work_path2}/

cd ${work_path2}

bash sdxl_export.sh
exit_code=$(($exit_code + $?))

bash sdxl_infer.sh
exit_code=$(($exit_code + $?))


echo exit_code:${exit_code}
exit ${exit_code}
