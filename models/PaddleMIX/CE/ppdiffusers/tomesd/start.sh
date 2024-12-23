#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/tomesd/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0

#bash prepare.sh




echo "*******tomesd begin***********"
(python tome.py) 2>&1 | tee ${log_dir}/tome.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "tome run success" >>"${log_dir}/ce_res.log"
else
    echo "tome run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******tome end***********"

echo "*******tomesd controlnet begin***********"
(python tome_controlnet.py) 2>&1 | tee ${log_dir}/tome_controlnet.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "tome controlnet run success" >>"${log_dir}/ce_res.log"
else
    echo "tome controlnet run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******tome controlnet end***********"




echo exit_code:${exit_code}
exit ${exit_code}
