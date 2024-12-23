#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/stable_video_diffusion/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0

bash prepare.sh



echo "*******stable_video_diffusion train begin***********"
(bash single_train.sh) 2>&1 | tee ${log_dir}/stable_video_diffusion_single_trian.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "stable_video_diffusion_single_train run success" >>"${log_dir}/ce_res.log"
else
    echo "stable_video_diffusion_single_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******stable_video_diffusion single_train end***********"


echo "*******stable_video_diffusion multi train begin***********"
(bash multi_train.sh) 2>&1 | tee ${log_dir}/stable_video_diffusion_multi_trian.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "stable_video_diffusion_multi_train run success" >>"${log_dir}/ce_res.log"
else
    echo "stable_video_diffusion_multi_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******stable_video_diffusion multi_train end***********"

echo "*******stable_video_diffusion inference begin***********"
(python inference.py) 2>&1 | tee ${log_dir}/stable_video_diffusion_inference.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "stable_video_diffusion_inference run success" >>"${log_dir}/ce_res.log"
else
    echo "stable_video_diffusion_inference run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******stable_video_diffusion inference end***********"

rm -rf ${work_path}/dataset/*
rm -rf ${work_path}/sky_timelapse_lvdm/*

echo exit_code:${exit_code}
exit ${exit_code}
