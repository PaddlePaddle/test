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
# bash prepare.sh
exit_code=0


export http_proxy=${mix_proxy}
export https_proxy=${mix_proxy}

export HF_ENDPOINT=https://hf-mirror.com
export no_proxy=baidu.com,127.0.0.1,0.0.0.0,localhost,bcebos.com,pip.baidu-int.com,mirrors.baidubce.com,repo.baidubce.com,repo.bcm.baidubce.com,pypi.tuna.tsinghua.edu.cn,aistudio.baidu.com
export USE_PPXFORMERS=true


# 安装额外的算子

cd ${work_path}/paddlemix/external_ops

echo "*******paddlemix ops_install begin begin***********"
(python setup.py install) 2>&1 | tee ${log_dir}/qwen2_vl_ops_install.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "qwen2_vl_ops_install run success" >>"${log_dir}/ut_res.log"
else
    echo "qwen2_vl_ops_install run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******paddlemix qwen2_vl_ops_install end***********"


cd ${work_path}


echo "*******paddlemix qwen2_vl_infer begin begin***********"
(python paddlemix/examples/qwen2_vl/single_image_infer.py) 2>&1 | tee ${log_dir}/qwen2_vl_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "qwen2_vl_infer run success" >>"${log_dir}/ut_res.log"
else
    echo "qwen2_vl_infer run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******paddlemix qwen2_vl_infer end***********"


echo "*******paddlemix qwen2_vl_multi_image_infer begin begin***********"
(python paddlemix/examples/qwen2_vl/multi_image_infer.py) 2>&1 | tee ${log_dir}/qwen2_vl_multi_image_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "qwen2_vl_multi_image_infer run success" >>"${log_dir}/ut_res.log"
else
    echo "qwen2_vl_multi_image_infer run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******paddlemix qwen2_vl_multi_image_infer end***********"


echo "*******paddlemix qwen2_vl_video begin begin***********"
(python paddlemix/examples/qwen2_vl/video_infer.py) 2>&1 | tee ${log_dir}/qwen2_vl_video.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "qwen2_vl_video run success" >>"${log_dir}/ut_res.log"
else
    echo "qwen2_vl_video run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******paddlemix qwen2_vl_video end***********"

# echo "*******paddlemix qwen2_vl_train begin begin***********"
# (bash train_qwen.sh) 2>&1 | tee ${log_dir}/qwen2_vl_train.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "qwen2_vl_train run success" >>"${log_dir}/ut_res.log"
# else
#     echo "qwen2_vl_train run fail" >>"${log_dir}/ut_res.log"
# fi
# echo "*******paddlemix qwen2_vl_train end***********"

# echo "*******paddlemix qwen2_vl_train_infer begin begin***********"
# (python iner_qwen.py) 2>&1 | tee ${log_dir}/qwen2_vl_train_infer.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "qwen2_vl_train_infer run success" >>"${log_dir}/ut_res.log"
# else
#     echo "qwen2_vl_train_infer run fail" >>"${log_dir}/ut_res.log"
# fi
echo "*******paddlemix qwen2_vl_train_infer end***********"
unset http_proxy
unset https_proxy


# # 查看结果
cat ${log_dir}/ut_res.log
echo exit_code:${exit_code}
exit ${exit_code}

