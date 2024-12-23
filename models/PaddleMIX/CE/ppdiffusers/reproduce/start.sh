#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/reproduce/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0


echo "*******reproduce infer begin***********"
(python infer.py) 2>&1 | tee ${log_dir}/reproduce_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "reproduce run success" >>"${log_dir}/ce_res.log"
else
    echo "reproduce run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******reproduce end***********"


echo exit_code:${exit_code}
exit ${exit_code}



