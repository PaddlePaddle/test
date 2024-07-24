#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/deploy/svd
echo ${work_path}

log_dir=${root_path}/deploy_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
exit_code=0

cd ${work_path}

bash export.sh
exit_code=$(($exit_code + $?))

bash infer.sh
exit_code=$(($exit_code + $?))

echo exit_code:${exit_code}
exit ${exit_code}
