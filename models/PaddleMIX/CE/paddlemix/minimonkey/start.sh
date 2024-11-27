#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/
echo ${work_path}

log_dir=${root_path}/paddlemix_examples_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ../change_paddlenlp_version.sh ${work_path}
/bin/cp -rf ./* ${work_path}
# cp ../change_paddlenlp_version.sh ${work_path}

cd ${work_path}
exit_code=0



export http_proxy=${mix_proxy}
export https_proxy=${mix_proxy}
# rm -rf tests/pipelines/test_pipelines.py
# rm -rf tests/pipelines/stable_diffusion/test_stable_diffusion_pix2pix_zero.py

exit_code=0

export HF_ENDPOINT=https://hf-mirror.com
export no_proxy=baidu.com,127.0.0.1,0.0.0.0,localhost,bcebos.com,pip.baidu-int.com,mirrors.baidubce.com,repo.baidubce.com,repo.bcm.baidubce.com,pypi.tuna.tsinghua.edu.cn,aistudio.baidu.com
export USE_PPXFORMERS=true

# bash prepare.sh
# 准备图片做物料
echo "*******paddlemix minimonkey_picture_infer begin begin***********"
cp ${work_path}/paddlemix/demo_images/examples_image1.jpg .

(python paddlemix/examples/minimonkey/chat_demo_minimonkey.py \
    --model_name_or_path "HUST-VLRLab/Mini-Monkey" \
    --image_path 'paddlemix/demo_images/examples_image1.jpg' \
    --text "Read the all text in the image.") 2>&1 | tee ${log_dir}/minimonkey_picture_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "minimonkey_picture_infer run success" >>"${log_dir}/ut_res.log"
else
    echo "minimonkey_picture_infer run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******paddlemix minimonkey_picture_infer end***********"


echo "*******paddlemix minimonkey_train begin begin***********"
# 只测2B模型即可 32G以下显存
(sh paddlemix/examples/minimonkey/shell/internvl2.0/2nd_finetune/minimonkey_2b_internlm2_1_8b_dynamic_res_2nd_finetune_full.sh) 2>&1 | tee ${log_dir}/minimonkey_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "minimonkey_train run success" >>"${log_dir}/ut_res.log"
else
    echo "minimonkey_train run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******paddlemix minimonkey_train end***********"


unset http_proxy
unset https_proxy

# rm -rf examples_image1.jpg
# rm -rf red-panda.mp4
# rm -rf playground
# 查看结果
cat ${log_dir}/ut_res.log

echo exit_code:${exit_code}
exit ${exit_code}
