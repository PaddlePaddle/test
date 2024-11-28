#!/bin/bash

log_dir=${root_path}/deploy_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

export FLAGS_use_cuda_managed_memory=true
export USE_PPXFORMERS=False

export FLAGS_allocator_strategy=auto_growth

export FLAGS_embedding_deterministic=1

export FLAGS_cudnn_deterministic=1

# text2img
(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-ipadapter/ \
    --scheduler "ddim" \
    --backend paddle \
    --device gpu \
    --task_name text2img) 2>&1 | tee ${log_dir}/ipadapter_sd15_inference_text2img.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_text2img success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_text2img fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_text2img end***********"

# img2img
(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-ipadapter/ \
    --scheduler "ddim" \
    --backend paddle \
    --device gpu \
    --task_name img2img) 2>&1 | tee ${log_dir}/ipadapter_sd15_inference_img2img.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_img2img success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_img2img fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_img2img end***********"

# inpaint
(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-ipadapter/ \
    --scheduler "ddim" \
    --backend paddle \
    --device gpu \
    --task_name inpaint_legacy) 2>&1 | tee ${log_dir}/ipadapter_sd15_inference_inpaint.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_inpaint success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_inpaint fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_inpaint end***********"

# tensorrt
# tune
(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-ipadapter/ \
    --scheduler "ddim" \
    --backend paddle \
    --device gpu \
    --task_name all \
    --width 512 \
    --height 512 \
    --inference_steps 5 \
    --tune True \
    --use_fp16 False) 2>&1 | tee ${log_dir}/ipadapter_sd15_inference_tune.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_tune success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_tune fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_tune end***********"

# text2img
(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-ipadapter/ \
    --scheduler "ddim" \
    --backend paddle_tensorrt \
    --device gpu \
    --width 512 \
    --height 512 \
    --inference_steps 50 \
    --task_name text2img) 2>&1 | tee ${log_dir}/ipadapter_sd15_inference_tensorrt_text2img.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_tensorrt_text2img success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_tensorrt_text2img fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_tensorrt_text2img end***********"

# img2img
(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-ipadapter/ \
    --scheduler "ddim" \
    --backend paddle_tensorrt \
    --device gpu \
    --width 512 \
    --height 512 \
    --inference_steps 50 \
    --task_name img2img) 2>&1 | tee ${log_dir}/ipadapter_sd15_inference_tensorrt_img2img.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_tensorrt_img2img success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_tensorrt_img2img fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_tensorrt_img2img end***********"

# inpaint
(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-ipadapter/ \
    --scheduler "ddim" \
    --backend paddle_tensorrt \
    --device gpu \
    --width 512 \
    --height 512 \
    --inference_steps 50 \
    --task_name inpaint_legacy) 2>&1 | tee ${log_dir}/ipadapter_sd15_inference_tensorrt_inpaint.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_tensorrt_inpaint success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_tensorrt_inpaint fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/ipadapter_sd15 ipadapter_sd15_inference_tensorrt_inpaint end***********"

echo exit_code:${exit_code}
exit ${exit_code}
