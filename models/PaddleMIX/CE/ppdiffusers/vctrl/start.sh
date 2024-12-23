#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/vctrl/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0

#bash prepare.sh




echo "*******vctrl canny2Video t2v begin***********"
(sh scripts/infer_cogvideox_t2v_vctrl.sh) 2>&1 | tee ${log_dir}/vctrl_canny2Video_t2v.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "vctrl_infer canny2Video t2v run success" >>"${log_dir}/ce_res.log"
else
    echo "vctrl_infer canny2Video t2v run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******vctrl infer canny2Video t2v end***********"

echo "*******vctrl canny2Video i2v begin***********"
(sh scripts/infer_cogvideox_i2v_vctrl.sh) 2>&1 | tee ${log_dir}/vctrl_canny2Video_i2v.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "vctrl_infer canny2Video i2v run success" >>"${log_dir}/ce_res.log"
else
    echo "vctrl_infer canny2Video i2v run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******vctrl infer canny2Video i2v end***********"


echo "*******vctrl pose2Video begin***********"
(sh scripts/infer_cogvideox_i2v_pose_vctrl.sh) 2>&1 | tee ${log_dir}/vctrl_pose2Video.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "vctrl_infer pose2Video run success" >>"${log_dir}/ce_res.log"
else
    echo "vctrl_infer pose2Video run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******vctrl infer pose2Video end***********"

echo "*******vctrl mask2Video t2v begin***********"
(sh infer_cogvideox_t2v_mask_vctrl.sh) 2>&1 | tee ${log_dir}/vctrl_pose2Video_t2v.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "vctrl_infer vctrl_pose2Video_t2v run success" >>"${log_dir}/ce_res.log"
else
    echo "vctrl_infer vctrl_pose2Video_t2v run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******vctrl infer vctrl_pose2Video_t2v end***********"

echo "*******vctrl mask2Video i2v begin***********"
(sh infer_cogvideox_i2v_mask_vctrl.sh) 2>&1 | tee ${log_dir}/vctrl_pose2Video_i2v.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "vctrl_infer vctrl_pose2Video_i2v run success" >>"${log_dir}/ce_res.log"
else
    echo "vctrl_infer vctrl_pose2Video_i2v run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******vctrl infer vctrl_pose2Video_i2v end***********"
# # 查看结果
# cat ${log_dir}/ce_res.log
rm -rf ${work_path}/ubcNbili_data/*

echo exit_code:${exit_code}
exit ${exit_code}
