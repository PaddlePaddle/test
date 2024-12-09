#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

#bash prepare.sh


work_path=${root_path}/PaddleMIX
echo ${work_path}
cp ${cur_path}/pp_data.json ${work_path}

log_dir=${root_path}/paddlemix_examples_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

cd ${work_path}
exit_code=0


export http_proxy=${mix_proxy}
export https_proxy=${mix_proxy}


exit_code=0

export HF_ENDPOINT=https://hf-mirror.com
export no_proxy=baidu.com,127.0.0.1,0.0.0.0,localhost,bcebos.com,pip.baidu-int.com,mirrors.baidubce.com,repo.baidubce.com,repo.bcm.baidubce.com,pypi.tuna.tsinghua.edu.cn,aistudio.baidu.com
export USE_PPXFORMERS=true


echo "*******paddlemix pp_inscaptagger_infer begin begin***********"
(python paddlemix/datacopilot/example/pp_inscaptagger/inference.py \
    single_data \
    -m paddlemix/PP-InsCapTagger \
    -image https://paddlenlp.bj.bcebos.com/models/community/paddlemix/PP-InsCapTagger/demo.jpg \
    -qa "What animal is in the image?" "The image features a dog." \
        "What color are the dog's eyes?" "The dog has blue eyes." \
        "Where is the dog situated in the image?" "The dog is situated inside a vehicle, on a front passenger seat.") 2>&1 | tee ${log_dir}/pp_inscaptagger_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "pp_inscaptagger_infer run success" >>"${log_dir}/ut_res.log"
else
    echo "pp_inscaptagger_infer run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******paddlemix pp_inscaptagger_infer end***********"

echo "*******paddlemix pp_inscaptagger_multi_infer begin begin***********"
rm -rf pp-output
(mkdir -p pp-output) 2>&1 | tee ${log_dir}/pp_inscaptagger_multi_infer.log
(python paddlemix/datacopilot/example/pp_inscaptagger/inference.py \
    json_data \
    -m paddlemix/PP-InsCapTagger \
    -d ./pp_data.json \
    -k 0 \
    -o pp-output) 2>&1 | tee ${log_dir}/pp_inscaptagger_multi_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "pp_inscaptagger_multi_infer run success" >>"${log_dir}/ut_res.log"
else
    echo "pp_inscaptagger_multi_infer run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******paddlemix pp_inscaptagger_multi_infer end***********"

unset http_proxy
unset https_proxy


# # 查看结果
cat ${log_dir}/ut_res.log
echo exit_code:${exit_code}
exit ${exit_code}

