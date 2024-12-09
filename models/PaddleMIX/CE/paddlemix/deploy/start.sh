#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

exit_code=0

# export http_proxy=${proxy}
# export https_proxy=${proxy}
# python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Linux-Gpu-Cuda11.7-Cudnn8-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post117-cp310-cp310-linux_x86_64.whl
# unset http_proxy
# unset https_proxy
# 即将下线
# bash blip2_deploy.sh
# exit_code=$(($exit_code + $?))

# python -m pip install paddlepaddle-gpu==2.5.1.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html -i http://pip.baidu.com/root/baidu/+simple/ --trusted-host pip.baidu.com

bash groundingdino_deploy.sh
exit_code=$(($exit_code + $?))

bash sam_deploy.sh
exit_code=$(($exit_code + $?))

bash qwen_deploy.sh
exit_code=$(($exit_code + $?))

# bash llava_deploy.sh
# exit_code=$(($exit_code + $?))
# # 查看结果
cat ${log_dir}/ce_res.log

echo exit_code:${exit_code}
exit ${exit_code}
