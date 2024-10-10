#!/bin/bash

log_dir=${root_path}/deploy_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

export FLAGS_use_cuda_managed_memory=true
export USE_PPXFORMERS=False

(python export_model.py --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
 --ipadapter_pretrained_model_name_or_path h94/IP-Adapter \
 --ipadapter_model_subfolder sdxl_models \
 --ipadapter_weight_name ip-adapter_sdxl.safetensors \
 --output_path static_model/stable-diffusion-xl-base-1.0-ipadapter) 2>&1 | tee ${log_dir}/ipadapter_sdxl_export_model.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sdxl ipadapter_sdxl_export_model success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sdxl ipadapter_sdxl_export_model fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sdxl ipadapter_sdxl_export_model end***********"

echo exit_code:${exit_code}
exit ${exit_code}
