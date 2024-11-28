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
# text2img
(python infer.py \
    --model_dir static_model/stable-video-diffusion-img2vid-xt \
    --scheduler "euler" \
    --backend paddle \
    --width 256 \
    --height 256 \
    --device gpu \
    --task_name img2video) 2>&1 | tee ${log_dir}/svd_inference_text2video.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/svd svd_inference_text2video success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/svd svd_inference_text2video fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/svd svd_inference_text2video end***********"


# tensorrt
#tune
(python infer.py \
    --model_dir static_model/stable-video-diffusion-img2vid-xt \
    --scheduler "euler" \
    --backend paddle \
    --device gpu \
    --task_name all \
    --width 256 \
    --height 256 \
    --inference_steps 5 \
    --tune True \
    --use_fp16 False) 2>&1 | tee ${log_dir}/svd_inference_tune.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/svd svd_inference_tune success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/svd svd_inference_tune fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/svd svd_inference_tune end***********"

# text2img
(python infer.py \
    --model_dir static_model/stable-video-diffusion-img2vid-xt \
    --scheduler "euler" \
    --backend paddle_tensorrt \
    --device gpu \
    --width 256 \
    --height 256 \
    --inference_steps 25 \
    --task_name img2video) 2>&1 | tee ${log_dir}/svd_inference_tensorrt_text2video.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/svd svd_inference_tensorrt_text2video success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/svd svd_inference_tensorrt_text2video fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/svd svd_inference_tensorrt_text2video end***********"

echo exit_code:${exit_code}
exit ${exit_code}
