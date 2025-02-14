#!/bin/bash

log_dir=${root_path}/deploy_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

export FLAGS_use_cuda_managed_memory=true
export USE_PPXFORMERS=False

(python export_model.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --ipadapter_pretrained_model_name_or_path h94/IP-Adapter \
    --ipadapter_model_subfolder models \
    --ipadapter_weight_name ip-adapter_sd15.safetensors \
    --output_path static_model/stable-diffusion-v1-5-ipadapter) 2>&1 | tee ${log_dir}/ipadapter_sd15_export_model.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sd15 ipadapter_sd15_export_model success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sd15 ipadapter_sd15_export_model fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sd15 ipadapter_sd15_export_model end***********"

echo exit_code:${exit_code}
exit ${exit_code}
