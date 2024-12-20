#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers
echo ${work_path}

log_dir=${root_path}/ut_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}
cd ${work_path}
exit_code=0




export http_proxy=${mix_proxy}
export https_proxy=${mix_proxy}
# rm -rf tests/pipelines/test_pipelines.py
# rm -rf tests/pipelines/stable_diffusion/test_stable_diffusion_pix2pix_zero.py

exit_code=0

export HF_ENDPOINT=https://hf-mirror.com
export no_proxy=baidu.com,127.0.0.1,0.0.0.0,localhost,bcebos.com,pip.baidu-int.com,mirrors.baidubce.com,repo.baidubce.com,repo.bcm.baidubce.com,pypi.tuna.tsinghua.edu.cn,aistudio.baidu.com
export USE_PPXFORMERS=True
export RUN_SLOW=True
# tests目前子目录 community models pipelines schedulers others community

echo "*******tests/schedulers begin***********"
(python -m pytest -v tests/schedulers) 2>&1 | tee ${log_dir}/tests_schedulers.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "tests/schedulers run success" >>"${log_dir}/ut_res.log"
else
    echo "tests/schedulers run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******tests/schedulers end***********"

echo "*******tests/others begin***********"
(python -m pytest -v tests/others) 2>&1 | tee ${log_dir}/tests_others.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "tests/others run success" >>"${log_dir}/ut_res.log"
else
    echo "tests/others run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******tests/others end***********"

echo "*******tests/models begin***********"
(python -m pytest -v tests/models) 2>&1 | tee ${log_dir}/tests_models.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "tests/models run success" >>"${log_dir}/ut_res.log"
else
    echo "tests/models run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******tests/models end***********"

pip install note-seq==0.0.5
echo "*******tests/pipelines begin***********"
(python -m pytest -v tests/pipelines) 2>&1 | tee ${log_dir}/tests_pipelines.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "tests/pipelines run success" >>"${log_dir}/ut_res.log"
else
    echo "tests/pipelines run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******tests/pipelines end***********"

echo "*******tests/community begin***********"
(python -m pytest -v tests/community) 2>&1 | tee ${log_dir}/tests_community.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "tests/community run success" >>"${log_dir}/ut_res.log"
else
    echo "tests/community run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******tests/community end***********"

unset http_proxy
unset https_proxy

# # 查看结果
cat ${log_dir}/ut_res.log

# 查看pip list 
pip list | grep paddle

pip list | grep huggingface
echo exit_code:${exit_code}
exit ${exit_code}
