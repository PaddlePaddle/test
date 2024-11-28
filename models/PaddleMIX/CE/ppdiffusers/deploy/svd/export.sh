#!/bin/bash

log_dir=${root_path}/deploy_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

export FLAGS_use_cuda_managed_memory=true
export USE_PPXFORMERS=False
export USE_PPXFORMERS=False

export FLAGS_allocator_strategy=auto_growth

export FLAGS_embedding_deterministic=1

export FLAGS_cudnn_deterministic=1
(python export_model.py \
    --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt \
    --output_path static_model/stable-video-diffusion-img2vid-xt) 2>&1 | tee ${log_dir}/deploy_svd_export_model.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sdxl sdxl_export_model success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sdxl sdxl_export_model fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sdxl sdxl_export_model end***********"

echo exit_code:${exit_code}
exit ${exit_code}
