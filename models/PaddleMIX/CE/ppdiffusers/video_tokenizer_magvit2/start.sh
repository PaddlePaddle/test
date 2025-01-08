#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/visual_tokenizer/magvit2/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0

# 下载依赖和数据
bash prepare.sh

echo "*******video_tokenizer/magvit2 begin***********"
(python example.py) 2>&1 | tee ${log_dir}/video_tokenizer_magvit2_test.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "video_tokenizer/magvit2 run success" >>"${log_dir}/ce_res.log"
else
    echo "video_tokenizer/magvit2 run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******video_tokenizer/magvit2 end***********"

echo exit_code:${exit_code}
exit ${exit_code}
