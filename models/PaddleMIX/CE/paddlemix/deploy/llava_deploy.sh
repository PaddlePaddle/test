#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/
echo ${work_path}

log_dir=${root_path}/log_deploy

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
exit_code=0

cd ${root_path}/PaddleMIX/
export http_proxy=${proxy}
export https_proxy=${proxy}

cd ${work_path}
# 通用
bash prepare_llava.sh

echo "*******paddlemix deploy ll av a begin***********"
cd ${work_path}

#静态图模型导出
(bash llava_export_vis.sh) 2>&1 | tee ${log_dir}/run_deploy_llava_export_vis.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix deploy llava export vis run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix deploy llava export vis run fail" >>"${log_dir}/ce_res.log"
fi
cd ${work_path}

#静态图模型导出 语言模型
(bash llava_export_lang.sh) 2>&1 | tee ${log_dir}/run_deploy_llava_export_lang.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix deploy llava export lang run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix deploy llava export lang run fail" >>"${log_dir}/ce_res.log"
fi
cd ${work_path}


# 预测
(bash llava_predict.sh) 2>&1 | tee ${log_dir}/run_deploy_llava_predict.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix deploy llava predict run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix deploy llava predict run fail" >>"${log_dir}/ce_res.log"
fi

echo "*******paddlemix deploy llava end***********"

# 检查命令是否成功执行
if [ ${exit_code} -ne 0 ]; then
    exit 1
fi
