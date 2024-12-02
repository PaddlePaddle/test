#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/inference/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
exit_code=0

cd ${work_path}

bash alt_diffusion_infer.sh
exit_code=$(($exit_code + $?))

bash audio_diffusion_infer.sh
exit_code=$(($exit_code + $?))

bash controlnet_infer.sh
exit_code=$(($exit_code + $?))

bash dance_diffusion_infer.sh
exit_code=$(($exit_code + $?))

bash ddim_infer.sh
exit_code=$(($exit_code + $?))

bash latent_diffusion_infer.sh
exit_code=$(($exit_code + $?))

bash paint_by_example_infer.sh
exit_code=$(($exit_code + $?))

# bash pndm_infer.sh
# exit_code=$(($exit_code + $?))

bash repaint_infer.sh
exit_code=$(($exit_code + $?))

# bash score_sde_ve_infer.sh
# exit_code=$(($exit_code + $?))

bash stable_diffusion_infer.sh
exit_code=$(($exit_code + $?))

bash stochastic_karras_ve_infer.sh
exit_code=$(($exit_code + $?))

bash unclip_infer.sh
exit_code=$(($exit_code + $?))

bash versatile_diffusion_infer.sh
exit_code=$(($exit_code + $?))

bash vq_diffusion_infer.sh
exit_code=$(($exit_code + $?))

# cat ${log_dir}/res.log

echo exit_code:${exit_code}
exit ${exit_code}
