#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/
echo ${work_path}

log_dir=${root_path}/paddlemix_examples_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

# /bin/cp -rf ../change_paddlenlp_version.sh ${work_path}
/bin/cp -rf ./* ${work_path}
/bin/cp -f ../check_loss.py ${work_path}
cd ${work_path}
exit_code=0



export http_proxy=${mix_proxy}
export https_proxy=${mix_proxy}


exit_code=0
export no_proxy=baidu.com,127.0.0.1,0.0.0.0,localhost,bcebos.com,pip.baidu-int.com,mirrors.baidubce.com,repo.baidubce.com,repo.bcm.baidubce.com,pypi.tuna.tsinghua.edu.cn,aistudio.baidu.com

bash prepare.sh
# 准备图片做物料
echo "*******paddlemix InternVL2_picture_infer begin begin***********"
cp ${work_path}/paddlemix/demo_images/examples_image1.jpg .

(python paddlemix/examples/internvl2/chat_demo.py \
    --model_name_or_path "OpenGVLab/InternVL2-8B" \
    --image_path 'paddlemix/demo_images/examples_image1.jpg' \
    --text "Please describe this image in detail.") 2>&1 | tee ${log_dir}/InternVL2_picture_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "InternVL2_picture_infer run success" >>"${log_dir}/ut_res.log"
else
    echo "InternVL2_picture_infer run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******paddlemix InternVL2_picture_infer end***********"


echo "*******paddlemix InternVL2_video_infer begin begin***********"
# 准备视频做物料
cp ${work_path}/paddlemix/demo_images/red-panda.mp4 .

(python paddlemix/examples/internvl2/chat_demo_video.py \
    --model_name_or_path "OpenGVLab/InternVL2-8B" \
    --video_path 'paddlemix/demo_images/red-panda.mp4' \
    --text "Please describe this video in detail.") 2>&1 | tee ${log_dir}/InternVL2_video_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "InternVL2_video_infer run success" >>"${log_dir}/ut_res.log"
else
    echo "InternVL2_video_infer run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******paddlemix InternVL2_video_infer end***********"


echo "*******paddlemix InternVL2_train begin begin***********"
# 只测2B模型即可 32G以下显存
(bash train_internvl2.sh) 2>&1 | tee ${log_dir}/InternVL2_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "InternVL2_train run success" >>"${log_dir}/ut_res.log"
else
    echo "InternVL2_train run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******paddlemix InternVL2_train end***********"

echo "*******paddlemix InternVL2_after_train_infer begin begin***********"
(python paddlemix/examples/internvl2/chat_demo.py \
    --model_name_or_path "work_dirs/internvl_chat_v2_0/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_full-2B" \
    --image_path 'paddlemix/demo_images/examples_image1.jpg' \
    --text "Please describe this image in detail.") 2>&1 | tee ${log_dir}/InternVL2_after_train_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "InternVL2_after_train_infer run success" >>"${log_dir}/ut_res.log"
else
    echo "InternVL2_after_train_infer run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******paddlemix InternVL2_after_train_infer end***********"

unset http_proxy
unset https_proxy

rm -rf examples_image1.jpg
rm -rf red-panda.mp4
# 查看结果
cat ${log_dir}/ut_res.log

echo exit_code:${exit_code}
exit ${exit_code}
