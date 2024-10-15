#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX
echo ${work_path}

log_dir=${root_path}/paddlemix_examples_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi


/bin/cp -rf ../change_paddlenlp_version.sh ${work_path}
/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0


export http_proxy=${mix_proxy}
export https_proxy=${mix_proxy}


exit_code=0

export HF_ENDPOINT=https://hf-mirror.com
export no_proxy=baidu.com,127.0.0.1,0.0.0.0,localhost,bcebos.com,pip.baidu-int.com,mirrors.baidubce.com,repo.baidubce.com,repo.bcm.baidubce.com,pypi.tuna.tsinghua.edu.cn,aistudio.baidu.com
export USE_PPXFORMERS=true


echo "*******paddlemix llava_next_interleave begin begin***********"
(python test_llava_next.py) 2>&1 | tee ${log_dir}/llava_next_interleave_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "llava_next_interleave_infer run success" >>"${log_dir}/ut_res.log"
else
    echo "llava_next_interleave_infer run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******paddlemix llava_next_interleave_infer end***********"



unset http_proxy
unset https_proxy


# # 查看结果
cat ${log_dir}/ut_res.log
echo exit_code:${exit_code}
exit ${exit_code}

