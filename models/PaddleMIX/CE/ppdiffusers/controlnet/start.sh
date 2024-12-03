#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/controlnet/
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

# 单机训练
echo "*******controlnet singe_train begin***********"
(bash single_train.sh) 2>&1 | tee ${log_dir}/controlnet_singe_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "controlnet singe_train run success" >>"${log_dir}/ce_res.log"
else
    echo "controlnet singe_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******controlnet singe_train end***********"

# 单机训练的结果进行推理
echo "******controlnet singe infer begin***********"
(python infer.py 2>&1) | tee ${log_dir}/controlnet_single_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "controlnet single_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "controlnet single_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******controlnet singe infer end***********"

# 多机训练
echo "*******controlnet muti_train begin***********"
(bash multi_train.sh) 2>&1 | tee ${log_dir}/controlnet_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "controlnet multi_train run success" >>"${log_dir}/ce_res.log"
else
    echo "controlnet multi_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******controlnet multi_train end***********"

# 多机训练的结果进行推理
echo "*******controlnet multi infer begin***********"
(python infer.py) 2>&1 | tee ${log_dir}/controlnet_multi_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "controlnet multi_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "controlnet multi_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******controlnet multi infer end***********"

# echo "*******controlnet gradio_canny2image begin***********"
# (python gradio_canny2image.py) 2>&1 | tee ${log_dir}/controlnet_gradio_canny2image.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "controlnet gradio_canny2image run success" >>"${log_dir}/ce_res.log"
# else
#     echo "controlnet gradio_canny2image run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******controlnet gradio_canny2image end***********"

# echo "*******controlnet gradio_hed2image begin***********"
# (python gradio_hed2image.py) 2>&1 | tee ${log_dir}/controlnet_gradio_hed2image.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "controlnet gradio_hed2image run success" >>"${log_dir}/ce_res.log"
# else
#     echo "controlnet gradio_hed2image run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******controlnet gradio_hed2image end***********"

# echo "*******controlnet gradio_pose2image begin***********"
# (python gradio_pose2image.py) 2>&1 | tee ${log_dir}/controlnet_gradio_hed2image.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "controlnet gradio_hed2image run success" >>"${log_dir}/ce_res.log"
# else
#     echo "controlnet gradio_hed2image run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******controlnet gradio_hed2image end***********"

# echo "*******controlnet gradio_seg2image_segmenter begin***********"
# (python gradio_seg2image_segmenter.py) 2>&1 | tee ${log_dir}/controlnet_gradio_seg2image_segmenter.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "controlnet gradio_seg2image_segmenter run success" >>"${log_dir}/ce_res.log"
# else
#     echo "controlnet gradio_seg2image_segmenter run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******controlnet gradio_seg2image_segmenter end***********"

# echo "*******controlnet gradio_seg2image_segmenter begin***********"
# (python gradio_seg2image_segmenter.py) 2>&1 | tee ${log_dir}/controlnet_gradio_seg2image_segmenter.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "controlnet gradio_seg2image_segmenter run success" >>"${log_dir}/ce_res.log"
# else
#     echo "controlnet gradio_seg2image_segmenter run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******controlnet gradio_seg2image_segmenter end***********"

# echo "*******controlnet gradio_depth2image begin***********"
# (python gradio_depth2image.py) 2>&1 | tee ${log_dir}/controlnet_gradio_depth2image.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "controlnet gradio_depth2image run success" >>"${log_dir}/ce_res.log"
# else
#     echo "controlnet gradio_depth2image run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******controlnet gradio_depth2image end***********"


# echo "*******controlnet gradio_normal2image begin***********"
# (python gradio_normal2image.py) 2>&1 | tee ${log_dir}/controlnet_gradio_normal2image.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "controlnet gradio_normal2image run success" >>"${log_dir}/ce_res.log"
# else
#     echo "controlnet gradio_normal2image run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******controlnet gradio_normal2image end***********"


# echo "*******controlnet gradio_hough2image begin***********"
# (python gradio_hough2image.py) 2>&1 | tee ${log_dir}/controlnet_gradio_hough2image.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "controlnet gradio_hough2image run success" >>"${log_dir}/ce_res.log"
# else
#     echo "controlnet gradio_hough2image run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******controlnet gradio_hough2image end***********"


# echo "*******controlnet gradio_ip2p2image begin***********"
# (python gradio_ip2p2image.py) 2>&1 | tee ${log_dir}/controlnet_gradio_ip2p2image.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "controlnet gradio_ip2p2image run success" >>"${log_dir}/ce_res.log"
# else
#     echo "controlnet gradio_ip2p2image run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******controlnet gradio_ip2p2image end***********"


# echo "*******controlnet gradio_shuffle2image begin***********"
# (python gradio_shuffle2image.py) 2>&1 | tee ${log_dir}/controlnet_gradio_shuffle2imagelog
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "controlnet gradio_shuffle2image run success" >>"${log_dir}/ce_res.log"
# else
#     echo "controlnet gradio_shuffle2image run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******controlnet gradio_shuffle2image end***********"
# # 查看结果
# cat ${log_dir}/ce_res.log
rm -rf ${work_path}/sd15_control/*
rm -rf ${work_path}/fill50k/

echo exit_code:${exit_code}
exit ${exit_code}
