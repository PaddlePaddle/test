#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/inference/
echo ${work_path}

work_path2=${root_path}/PaddleMIX/ppdiffusers/
echo ${work_path}

log_dir=${root_path}/infer_log


if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}


pip install pytest safetensors ftfy fastcore opencv-python einops parameterized requests-mock
pip install ligo-segments
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html

cd ${work_path}
exit_code=0

export FLAGS_use_cuda_managed_memory=true
export FLAGS_allocator_strategy=auto_growth
export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1

echo "*******infer dual_text_and_image_guided_generation-versatile_diffusion begin***********"
(python dual_text_and_image_guided_generation-versatile_diffusion.py) 2>&1 | tee ${log_dir}/dual_text_and_image_guided_generation-versatile_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer dual_text_and_image_guided_generation-versatile_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer dual_text_and_image_guided_generation-versatile_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer dual_text_and_image_guided_generation-versatile_diffusion end***********"


echo "*******infer image_guided_image_inpainting-paint_by_example begin***********"
(python image_guided_image_inpainting-paint_by_example.py) 2>&1 | tee ${log_dir}/image_guided_image_inpainting-paint_by_example.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_guided_image_inpainting-paint_by_example run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_guided_image_inpainting-paint_by_example run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_guided_image_inpainting-paint_by_example end***********"

echo "*******infer image_inpainting-repaint begin***********"
(python image_inpainting-repaint.py) 2>&1 | tee ${log_dir}/image_inpainting-repaint.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_inpainting-repaint run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_inpainting-repaint run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_inpainting-repaint end***********"


echo "*******infer image_mixing-clip_guided_stable_diffusion begin***********"
(python image_mixing-clip_guided_stable_diffusion.py) 2>&1 | tee ${log_dir}/image_mixing-clip_guided_stable_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_mixing-clip_guided_stable_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_mixing-clip_guided_stable_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_mixing-clip_guided_stable_diffusion end***********"


echo "*******infer image_to_image_text_guided_generation-alt_diffusion begin***********"
(python image_to_image_text_guided_generation-alt_diffusion.py) 2>&1 | tee ${log_dir}/image_to_image_text_guided_generation-alt_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_to_image_text_guided_generation-alt_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_to_image_text_guided_generation-alt_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_to_image_text_guided_generation-alt_diffusion end***********"


echo "*******infer image_to_image_text_guided_generation-controlnet begin***********"
(python image_to_image_text_guided_generation-controlnet.py) 2>&1 | tee ${log_dir}/image_to_image_text_guided_generation-controlnet.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_to_image_text_guided_generation-controlnet run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_to_image_text_guided_generation-controlnet run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_to_image_text_guided_generation-controlnet end***********"


echo "*******infer image_to_image_text_guided_generation-deepfloyd_if begin***********"
(python image_to_image_text_guided_generation-deepfloyd_if.py) 2>&1 | tee ${log_dir}/image_to_image_text_guided_generation-deepfloyd_if.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_to_image_text_guided_generation-deepfloyd_if run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_to_image_text_guided_generation-deepfloyd_if run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_to_image_text_guided_generation-deepfloyd_if end***********"


echo "*******infer image_to_image_text_guided_generation-kandinsky2_2_controlnet begin***********"
(python image_to_image_text_guided_generation-kandinsky2_2_controlnet.py) 2>&1 | tee ${log_dir}/image_to_image_text_guided_generation-kandinsky2_2_controlnet.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_to_image_text_guided_generation-kandinsky2_2_controlnet run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_to_image_text_guided_generation-kandinsky2_2_controlnet run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_to_image_text_guided_generation-kandinsky2_2_controlnet end***********"


echo "*******infer image_to_image_text_guided_generation-kandinsky2_2 begin***********"
(python image_to_image_text_guided_generation-kandinsky2_2.py) 2>&1 | tee ${log_dir}/image_to_image_text_guided_generation-kandinsky2_2.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_to_image_text_guided_generation-kandinsky2_2 run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_to_image_text_guided_generation-kandinsky2_2 run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_to_image_text_guided_generation-kandinsky2_2 end***********"


echo "*******infer image_to_image_text_guided_generation-kandinsky begin***********"
(python image_to_image_text_guided_generation-kandinsky.py) 2>&1 | tee ${log_dir}/image_to_image_text_guided_generation-kandinsky.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_to_image_text_guided_generation-kandinsky run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_to_image_text_guided_generation-kandinsky run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_to_image_text_guided_generation-kandinsky end***********"


echo "*******infer image_to_image_text_guided_generation-stable_diffusion_2 begin***********"
(python image_to_image_text_guided_generation-stable_diffusion_2.py) 2>&1 | tee ${log_dir}/image_to_image_text_guided_generation-stable_diffusion_2.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_to_image_text_guided_generation-stable_diffusion_2 run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_to_image_text_guided_generation-stable_diffusion_2 run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_to_image_text_guided_generation-stable_diffusion_2 end***********"

echo "*******infer image_to_image_text_guided_generation-stable_diffusion_3 begin***********"
(python image_to_image_text_guided_generation-stable_diffusion_3.py) 2>&1 | tee ${log_dir}/image_to_image_text_guided_generation-stable_diffusion_3.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_to_image_text_guided_generation-stable_diffusion_3 run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_to_image_text_guided_generation-stable_diffusion_3 run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_to_image_text_guided_generation-stable_diffusion_3 end***********"


echo "*******infer image_to_image_text_guided_generation-stable_diffusion_controlnet begin***********"
(python image_to_image_text_guided_generation-stable_diffusion_controlnet.py) 2>&1 | tee ${log_dir}/image_to_image_text_guided_generation-stable_diffusion_controlnet.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_to_image_text_guided_generation-stable_diffusion_controlnet run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_to_image_text_guided_generation-stable_diffusion_controlnet run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_to_image_text_guided_generation-stable_diffusion_controlnet end***********"


echo "*******infer image_to_image_text_guided_generation-stable_diffusion begin***********"
(python image_to_image_text_guided_generation-stable_diffusion.py) 2>&1 | tee ${log_dir}/image_to_image_text_guided_generation-stable_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_to_image_text_guided_generation-stable_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_to_image_text_guided_generation-stable_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_to_image_text_guided_generation-stable_diffusion end***********"


echo "*******infer image_to_image_text_guided_generation-stable_diffusion_xl begin***********"
(python image_to_image_text_guided_generation-stable_diffusion_xl.py) 2>&1 | tee ${log_dir}/image_to_image_text_guided_generation-stable_diffusion_xl.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_to_image_text_guided_generation-stable_diffusion_xl run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_to_image_text_guided_generation-stable_diffusion_xl run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_to_image_text_guided_generation-stable_diffusion_xl end***********"


echo "*******infer image_to_text_generation-unidiffuser begin***********"
(python image_to_text_generation-unidiffuser.py) 2>&1 | tee ${log_dir}/image_to_text_generation-unidiffuser.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_to_text_generation-unidiffuser run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_to_text_generation-unidiffuser run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_to_text_generation-unidiffuser end***********"


echo "*******infer image_variation-stable_diffusion begin***********"
(python image_variation-stable_diffusion.py) 2>&1 | tee ${log_dir}/image_variation-stable_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_variation-stable_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_variation-stable_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_variation-stable_diffusion end***********"


echo "*******infer image_variation-unidiffuser begin***********"
(python image_variation-unidiffuser.py) 2>&1 | tee ${log_dir}/image_variation-unidiffuser.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_variation-unidiffuser run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_variation-unidiffuser run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_variation-unidiffuser end***********"


echo "*******infer image_variation-versatile_diffusion begin***********"
(python image_variation-versatile_diffusion.py) 2>&1 | tee ${log_dir}/image_variation-versatile_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_variation-versatile_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_variation-versatile_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_variation-versatile_diffusion end***********"


echo "*******infer instruct_pix2pix-stable_diffusion_xl begin***********"
(python instruct_pix2pix-stable_diffusion_xl.py) 2>&1 | tee ${log_dir}/instruct_pix2pix-stable_diffusion_xl.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer instruct_pix2pix-stable_diffusion_xl run success" >>"${log_dir}/infer_res.log"
else
    echo "infer instruct_pix2pix-stable_diffusion_xl run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer instruct_pix2pix-stable_diffusion_xl end***********"


echo "*******infer super_resolution-latent_diffusion begin***********"
(python super_resolution-latent_diffusion.py) 2>&1 | tee ${log_dir}/super_resolution-latent_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer super_resolution-latent_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer super_resolution-latent_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer super_resolution-latent_diffusion end***********"


echo "*******infer text_guided_generation-semantic_stable_diffusion begin***********"
(python text_guided_generation-semantic_stable_diffusion.py) 2>&1 | tee ${log_dir}/text_guided_generation-semantic_stable_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_guided_generation-semantic_stable_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_guided_generation-semantic_stable_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_guided_generation-semantic_stable_diffusion end***********"


echo "*******infer text_guided_generation-semantic_stable_diffusion begin***********"
(python text_guided_image_inpainting-deepfloyd_if.py) 2>&1 | tee ${log_dir}/text_guided_image_inpainting-deepfloyd_if.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_guided_image_inpainting-deepfloyd_if run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_guided_image_inpainting-deepfloyd_if run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_guided_image_inpainting-deepfloyd_if end***********"


echo "*******infer text_guided_image_inpainting-kandinsky2_2 begin***********"
(python text_guided_image_inpainting-kandinsky2_2.py) 2>&1 | tee ${log_dir}/text_guided_image_inpainting-kandinsky2_2.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_guided_image_inpainting-kandinsky2_2 run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_guided_image_inpainting-kandinsky2_2 run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_guided_image_inpainting-kandinsky2_2 end***********"


echo "*******infer text_guided_image_inpainting-kandinsky begin***********"
(python text_guided_image_inpainting-kandinsky.py) 2>&1 | tee ${log_dir}/text_guided_image_inpainting-kandinsky.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_guided_image_inpainting-kandinsky run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_guided_image_inpainting-kandinsky run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_guided_image_inpainting-kandinsky end***********"


echo "*******infer text_guided_image_inpainting-stable_diffusion_2 begin***********"
(python text_guided_image_inpainting-stable_diffusion_2.py) 2>&1 | tee ${log_dir}/text_guided_image_inpainting-stable_diffusion_2.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_guided_image_inpainting-stable_diffusion_2 run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_guided_image_inpainting-stable_diffusion_2 run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_guided_image_inpainting-stable_diffusion_2 end***********"


echo "*******infer text_guided_image_inpainting-stable_diffusion_controlnet begin***********"
(python text_guided_image_inpainting-stable_diffusion_controlnet.py) 2>&1 | tee ${log_dir}/text_guided_image_inpainting-stable_diffusion_controlnet.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_guided_image_inpainting-stable_diffusion_controlnet run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_guided_image_inpainting-stable_diffusion_controlnet run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_guided_image_inpainting-stable_diffusion_controlnet end***********"


echo "*******infer text_guided_image_inpainting-stable_diffusion begin***********"
(python text_guided_image_inpainting-stable_diffusion.py) 2>&1 | tee ${log_dir}/text_guided_image_inpainting-stable_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_guided_image_inpainting-stable_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_guided_image_inpainting-stable_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_guided_image_inpainting-stable_diffusion end***********"


echo "*******infer text_guided_image_inpainting-stable_diffusion_xl begin***********"
(python text_guided_image_inpainting-stable_diffusion_xl.py) 2>&1 | tee ${log_dir}/text_guided_image_inpainting-stable_diffusion_xl.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_guided_image_inpainting-stable_diffusion_xl run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_guided_image_inpainting-stable_diffusion_xl run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_guided_image_inpainting-stable_diffusion_xl end***********"


echo "*******infer text_guided_image_upscaling-stable_diffusion_2 begin***********"
(python text_guided_image_upscaling-stable_diffusion_2.py) 2>&1 | tee ${log_dir}/text_guided_image_upscaling-stable_diffusion_2.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_guided_image_upscaling-stable_diffusion_2 run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_guided_image_upscaling-stable_diffusion_2 run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_guided_image_upscaling-stable_diffusion_2 end***********"


echo "*******infer text_to_3d_generation-shape_e begin***********"
(python text_to_3d_generation-shape_e.py) 2>&1 | tee ${log_dir}/text_to_3d_generation-shape_e.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_3d_generation-shape_e run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_3d_generation-shape_e run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_3d_generation-shape_eend***********"


echo "*******infer text_to_3d_generation-shape_e_image2image begin***********"
(python text_to_3d_generation-shape_e_image2image.py) 2>&1 | tee ${log_dir}/text_to_3d_generation-shape_e_image2image.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_3d_generation-shape_e_image2image run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_3d_generation-shape_e_image2image run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_3d_generation-shape_e_image2image end***********"


echo "*******infer text_to_audio_generation-audio_ldm begin***********"
(python text_to_audio_generation-audio_ldm.py) 2>&1 | tee ${log_dir}/text_to_audio_generation-audio_ldm.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_audio_generation-audio_ldm run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_audio_generation-audio_ldm run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_audio_generation-audio_ldm end***********"


echo "*******infer text_to_image_generation-alt_diffusion begin***********"
(python text_to_image_generation-alt_diffusion.py) 2>&1 | tee ${log_dir}/text_to_image_generation-alt_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-alt_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-alt_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-alt_diffusion end***********"


echo "*******infer text_to_image_generation-auto begin***********"
(python text_to_image_generation-auto.py) 2>&1 | tee ${log_dir}/text_to_image_generation-auto.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-auto run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-auto run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-auto end***********"


echo "*******infer text_to_image_generation-consistency_models begin***********"
(python text_to_image_generation-consistency_models.py) 2>&1 | tee ${log_dir}/text_to_image_generation-consistency_models.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-consistency_models run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-consistency_models run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-consistency_models end***********"


echo "*******infer text_to_image_generation-deepfloyd_if begin***********"
(python text_to_image_generation-deepfloyd_if.py) 2>&1 | tee ${log_dir}/text_to_image_generation-deepfloyd_if.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-deepfloyd_if run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-deepfloyd_if run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-deepfloyd_if end***********"


echo "*******infer text_to_image_generation-kandinsky2_2_controlnet begin***********"
(python text_to_image_generation-kandinsky2_2_controlnet.py) 2>&1 | tee ${log_dir}/text_to_image_generation-kandinsky2_2_controlnet.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-kandinsky2_2_controlnet run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-kandinsky2_2_controlnet run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-kandinsky2_2_controlnet end***********"


echo "*******infer text_to_image_generation-kandinsky2_2 begin***********"
(python text_to_image_generation-kandinsky2_2.py) 2>&1 | tee ${log_dir}/text_to_image_generation-kandinsky2_2.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-kandinsky2_2 run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-kandinsky2_2 run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-kandinsky2_2 end***********"


echo "*******infer text_to_image_generation-kandinsky begin***********"
(python text_to_image_generation-kandinsky.py) 2>&1 | tee ${log_dir}/text_to_image_generation-kandinsky.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-kandinsky run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-kandinsky run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-kandinsky end***********"


echo "*******infer text_to_image_generation-latent_diffusion begin***********"
(python text_to_image_generation-latent_diffusion.py) 2>&1 | tee ${log_dir}/text_to_image_generation-latent_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-latent_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-latent_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-latent_diffusion end***********"


echo "*******infer text_to_image_generation_mixture_tiling-stable_diffusion begin***********"
(python text_to_image_generation_mixture_tiling-stable_diffusion.py) 2>&1 | tee ${log_dir}/text_to_image_generation_mixture_tiling-stable_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation_mixture_tiling-stable_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation_mixture_tiling-stable_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation_mixture_tiling-stable_diffusion end***********"


echo "*******infer text_to_image_generation-sdxl_base_with_refiner begin***********"
(python text_to_image_generation-sdxl_base_with_refiner.py) 2>&1 | tee ${log_dir}/text_to_image_generation-sdxl_base_with_refiner.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-sdxl_base_with_refiner run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-sdxl_base_with_refiner run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-sdxl_base_with_refiner end***********"


echo "*******infer text_to_image_generation-stable_diffusion_2 begin***********"
(python text_to_image_generation-stable_diffusion_2.py) 2>&1 | tee ${log_dir}/text_to_image_generation-stable_diffusion_2.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-stable_diffusion_2 run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-stable_diffusion_2 run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-stable_diffusion_2 end***********"


echo "*******infer text_to_image_generation-stable_diffusion_3 begin***********"
(python text_to_image_generation-stable_diffusion_3.py) 2>&1 | tee ${log_dir}/text_to_image_generation-stable_diffusion_3.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-stable_diffusion_3 run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-stable_diffusion_3 run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-stable_diffusion_3 end***********"


echo "*******infer text_to_image_generation-stable_diffusion_controlnet begin***********"
(python text_to_image_generation-stable_diffusion_controlnet.py) 2>&1 | tee ${log_dir}/text_to_image_generation-stable_diffusion_controlnet.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-stable_diffusion_controlnet run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-stable_diffusion_controlnet run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-stable_diffusion_controlnet end***********"


echo "*******infer text_to_image_generation-stable_diffusion begin***********"
(python text_to_image_generation-stable_diffusion.py) 2>&1 | tee ${log_dir}/text_to_image_generation-stable_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-stable_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-stable_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-stable_diffusion end***********"


echo "*******infer text_to_image_generation-stable_diffusion_safe begin***********"
(python text_to_image_generation-stable_diffusion_safe.py) 2>&1 | tee ${log_dir}/text_to_image_generation-stable_diffusion_safe.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-stable_diffusion_safe run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-stable_diffusion_safe run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-stable_diffusion_safe end***********"


echo "*******infer text_to_image_generation-stable_diffusion_t2i_adapter begin***********"
(python text_to_image_generation-stable_diffusion_t2i_adapter.py) 2>&1 | tee ${log_dir}/text_to_image_generation-stable_diffusion_t2i_adapter.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-stable_diffusion_t2i_adapter run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-stable_diffusion_t2i_adapter run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-stable_diffusion_t2i_adapter end***********"


echo "*******infer text_to_image_generation-stable_diffusion_xl_controlnet begin***********"
(python text_to_image_generation-stable_diffusion_xl_controlnet.py) 2>&1 | tee ${log_dir}/text_to_image_generation-stable_diffusion_xl_controlnet.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-stable_diffusion_xl_controlnet run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-stable_diffusion_xl_controlnet run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-stable_diffusion_xl_controlnet end***********"


echo "*******infer text_to_image_generation-stable_diffusion_xl begin***********"
(python text_to_image_generation-stable_diffusion_xl.py) 2>&1 | tee ${log_dir}/text_to_image_generation-stable_diffusion_xl.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-stable_diffusion_xl run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-stable_diffusion_xl run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-stable_diffusion_xl end***********"


echo "*******infer text_to_image_generation-t2i-adapter begin***********"
(python text_to_image_generation-t2i-adapter.py) 2>&1 | tee ${log_dir}/text_to_image_generation-t2i-adapter.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-t2i-adapter run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-t2i-adapter run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-t2i-adapter end***********"


echo "*******infer text_to_image_generation-unclip begin***********"
(python text_to_image_generation-unclip.py) 2>&1 | tee ${log_dir}/text_to_image_generation-unclip.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-unclip run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-unclip run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-unclip end***********"


echo "*******infer text_to_image_generation-unidiffuser begin***********"
(python text_to_image_generation-unidiffuser.py) 2>&1 | tee ${log_dir}/text_to_image_generation-unidiffuser.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-unidiffuser run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-unidiffuser run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-unidiffuser end***********"


echo "*******infer text_to_image_generation-versatile_diffusion begin***********"
(python text_to_image_generation-versatile_diffusion.py) 2>&1 | tee ${log_dir}/text_to_image_generation-versatile_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-versatile_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-versatile_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-versatile_diffusion end***********"


echo "*******infer text_to_image_generation-vq_diffusion begin***********"
(python text_to_image_generation-vq_diffusion.py) 2>&1 | tee ${log_dir}/text_to_image_generation-vq_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-vq_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-vq_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-vq_diffusion end***********"


echo "*******infer text_to_video_generation-lvdm begin***********"
(python text_to_video_generation-lvdm.py) 2>&1 | tee ${log_dir}/text_to_video_generation-lvdm.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_video_generation-lvdm run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_video_generation-lvdm run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_video_generation-lvdm end***********"


echo "*******infer text_to_video_generation-synth_img2img begin***********"
(python text_to_video_generation-synth_img2img.py) 2>&1 | tee ${log_dir}/text_to_video_generation-synth_img2img.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_video_generation-synth_img2img run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_video_generation-synth_img2img run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_video_generation-synth_img2img end***********"


echo "*******infer text_to_video_generation-synth begin***********"
(python text_to_video_generation-synth.py) 2>&1 | tee ${log_dir}/text_to_video_generation-synth.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_video_generation-synth run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_video_generation-synth run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_video_generation-synth end***********"


echo "*******infer text_to_video_generation-zero begin***********"
(python text_to_video_generation-zero.py) 2>&1 | tee ${log_dir}/text_to_video_generation-zero.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_video_generation-zero run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_video_generation-zero run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_video_generation-zero end***********"


echo "*******infer text_variation-unidiffuser begin***********"
(python text_variation-unidiffuser.py) 2>&1 | tee ${log_dir}/text_variation-unidiffuser.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_variation-unidiffuser run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_variation-unidiffuser run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_variation-unidiffuser end***********"


echo "*******infer unconditional_audio_generation-audio_diffusion begin***********"
(python unconditional_audio_generation-audio_diffusion.py) 2>&1 | tee ${log_dir}/unconditional_audio_generation-audio_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_audio_generation-audio_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_audio_generation-audio_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer unconditional_audio_generation-audio_diffusion end***********"


echo "*******infer unconditional_audio_generation-dance_diffusion begin***********"
(python unconditional_audio_generation-dance_diffusion.py) 2>&1 | tee ${log_dir}/unconditional_audio_generation-dance_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_audio_generation-dance_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_audio_generation-dance_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer unconditional_audio_generation-dance_diffusion end***********"


echo "*******infer unconditional_audio_generation-spectrogram_diffusion begin***********"
(python unconditional_audio_generation-spectrogram_diffusion.py) 2>&1 | tee ${log_dir}/unconditional_audio_generation-spectrogram_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_audio_generation-spectrogram_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_audio_generation-spectrogram_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer unconditional_audio_generation-spectrogram_diffusion end***********"


echo "*******infer unconditional_image_generation-ddim begin***********"
(python unconditional_image_generation-ddim.py) 2>&1 | tee ${log_dir}/unconditional_image_generation-ddim.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_image_generation-ddim run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_image_generation-ddim run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer unconditional_image_generation-ddim end***********"


echo "*******infer unconditional_image_generation-ddpm begin***********"
(python unconditional_image_generation-ddpm.py) 2>&1 | tee ${log_dir}/unconditional_image_generation-ddpm.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_image_generation-ddpm run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_image_generation-ddpm run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer unconditional_image_generation-ddpm end***********"


echo "*******infer unconditional_image_generation-latent_diffusion_uncond begin***********"
(python unconditional_image_generation-latent_diffusion_uncond.py) 2>&1 | tee ${log_dir}/unconditional_image_generation-latent_diffusion_uncond.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_image_generation-latent_diffusion_uncond run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_image_generation-latent_diffusion_uncond run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer unconditional_image_generation-latent_diffusion_uncond end***********"


echo "*******infer unconditional_image_generation-pndm begin***********"
(python unconditional_image_generation-pndm.py) 2>&1 | tee ${log_dir}/unconditional_image_generation-pndm.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_image_generation-pndm run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_image_generation-pndm run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer unconditional_image_generation-pndm end***********"


echo "*******infer unconditional_image_generation-score_sde_ve begin***********"
(python unconditional_image_generation-score_sde_ve.py) 2>&1 | tee ${log_dir}/unconditional_image_generation-score_sde_ve.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_image_generation-score_sde_ve run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_image_generation-score_sde_ve run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer unconditional_image_generation-score_sde_ve end***********"


echo "*******infer unconditional_image_generation-stochastic_karras_ve begin***********"
(python unconditional_image_generation-stochastic_karras_ve.py) 2>&1 | tee ${log_dir}/unconditional_image_generation-stochastic_karras_ve.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_image_generation-stochastic_karras_ve run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_image_generation-stochastic_karras_ve run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer unconditional_image_generation-stochastic_karras_ve end***********"


echo "*******infer unconditional_image_text_joint_generation-unidiffuser begin***********"
(python unconditional_image_text_joint_generation-unidiffuser.py) 2>&1 | tee ${log_dir}/unconditional_image_text_joint_generation-unidiffuser.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_image_text_joint_generation-unidiffuser run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_image_text_joint_generation-unidiffuser run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer unconditional_image_text_joint_generation-unidiffuser end***********"

echo "*******infer class_conditional_image_generation-dit begin***********"
(python class_conditional_image_generation-dit.py) 2>&1 | tee ${log_dir}/class_conditional_image_generation-dit.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer class_conditional_image_generation-dit run success" >>"${log_dir}/infer_res.log"
else
    echo "infer class_conditional_image_generation-dit run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer class_conditional_image_generation-dit end***********"


echo "*******infer class_conditional_image_generation-large_dit_3b begin***********"
(python class_conditional_image_generation-large_dit_3b.py) 2>&1 | tee ${log_dir}/class_conditional_image_generation-large_dit_3b.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer class_conditional_image_generation-large_dit_3b run success" >>"${log_dir}/infer_res.log"
else
    echo "infer class_conditional_image_generation-large_dit_3b run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer class_conditional_image_generation-large_dit_3b end***********"


echo "*******infer class_conditional_image_generation-large_dit_7b begin***********"
(python class_conditional_image_generation-large_dit_7b.py) 2>&1 | tee ${log_dir}/class_conditional_image_generation-large_dit_7b.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer class_conditional_image_generation-large_dit_7b run success" >>"${log_dir}/infer_res.log"
else
    echo "infer class_conditional_image_generation-large_dit_7b run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer class_conditional_image_generation-large_dit_7b end***********"


echo "*******infer image_to_video_generation_image_to_video begin***********"
(python image_to_video_generation_image_to_video.py) 2>&1 | tee ${log_dir}/image_to_video_generation_image_to_video.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_to_video_generation_image_to_video run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_to_video_generation_image_to_video run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_to_video_generation_image_to_video end***********"


echo "*******infer image_to_video_generation_stable_video_diffusion begin***********"
(python image_to_video_generation_stable_video_diffusion.py) 2>&1 | tee ${log_dir}/image_to_video_generation_stable_video_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_to_video_generation_stable_video_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_to_video_generation_stable_video_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer image_to_video_generation_stable_video_diffusion end***********"


echo "*******infer text_to_audio_generation-audio_ldm2 begin***********"
(python text_to_audio_generation-audio_ldm2.py) 2>&1 | tee ${log_dir}/text_to_audio_generation-audio_ldm2.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_audio_generation-audio_ldm2 run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_audio_generation-audio_ldm2 run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_audio_generation-audio_ldm2 end***********"

echo "*******infer text_to_image_generation-latent_diffusion_uvit_small begin***********"
(python text_to_image_generation-latent_diffusion_uvit_small.py) 2>&1 | tee ${log_dir}/text_to_image_generation-latent_diffusion_uvit_small.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-latent_diffusion_uvit_small run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-latent_diffusion_uvit_small run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-latent_diffusion_uvit_small end***********"

echo "*******infer text_to_image_generation_consistency_model begin***********"
(python text_to_image_generation_consistency_model.py) 2>&1 | tee ${log_dir}/text_to_image_generation_consistency_model.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation_consistency_model run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation_consistency_model run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation_consistency_model end***********"


echo "*******infer text_to_image_generation_kandinsky3 begin***********"
(python text_to_image_generation_kandinsky3.py) 2>&1 | tee ${log_dir}/text_to_image_generation_kandinsky3.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation_kandinsky3 run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation_kandinsky3 run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation_kandinsky3 end***********"

echo "*******infer text_to_image_generation_largedit_3b begin***********"
(python text_to_image_generation_largedit_3b.py) 2>&1 | tee ${log_dir}/text_to_image_generation_largedit_3b.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation_largedit_3b run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation_largedit_3b run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation_largedit_3b end***********"

echo "*******infer text_to_image_generation_wuerstchen begin***********"
(python text_to_image_generation_wuerstchen.py) 2>&1 | tee ${log_dir}/text_to_image_generation_wuerstchen.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation_wuerstchen run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation_wuerstchen run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation_wuerstchen end***********"


echo "*******infer text_to_video_generation-synth begin***********"
(python text_to_video_generation-synth.py) 2>&1 | tee ${log_dir}/text_to_video_generation-synth.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_video_generation-synth run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_video_generation-synth run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_video_generation-synth end***********"


echo "*******infer text_to_video_generation-synth_img2img begin***********"
(python text_to_video_generation-synth_img2img.py) 2>&1 | tee ${log_dir}/text_to_video_generation-synth_img2img.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_video_generation-synth_img2img run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_video_generation-synth_img2img run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_video_generation-synth_img2img end***********"

echo "*******infer text_to_video_generation-zero begin***********"
(python text_to_video_generation-zero.py) 2>&1 | tee ${log_dir}/text_to_video_generation-zero.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_video_generation-zero run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_video_generation-zero run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_video_generation-zero end***********"


echo "*******infer text_to_video_generation_animediff begin***********"
(python text_to_video_generation_animediff.py) 2>&1 | tee ${log_dir}/text_to_video_generation_animediff.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_video_generation_animediff run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_video_generation_animediff run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_video_generation_animediff end***********"

echo "*******infer unconditional_audio_generation-audio_diffusion begin***********"
(python unconditional_audio_generation-audio_diffusion.py) 2>&1 | tee ${log_dir}/unconditional_audio_generation-audio_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_audio_generation-audio_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_audio_generation-audio_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer unconditional_audio_generation-audio_diffusion end***********"


echo "*******infer unconditional_audio_generation-dance_diffusion begin***********"
(python unconditional_audio_generation-dance_diffusion.py) 2>&1 | tee ${log_dir}/unconditional_audio_generation-dance_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_audio_generation-dance_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_audio_generation-dance_diffusion run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer unconditional_audio_generation-dance_diffusion end***********"


echo "*******infer video_to_video_generation_video_to_video begin***********"
(python video_to_video_generation_video_to_video.py) 2>&1 | tee ${log_dir}/video_to_video_generation_video_to_video.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer video_to_video_generation_video_to_video run success" >>"${log_dir}/infer_res.log"
else
    echo "infer video_to_video_generation_video_to_video run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer video_to_video_generation_video_to_video end***********"

echo "*******infer text_to_image_generation-stable_diffusion_3_5.py begin***********"
(python text_to_image_generation-stable_diffusion_3_5.py) 2>&1 | tee ${log_dir}/text_to_image_generation-stable_diffusion_3_5.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-stable_diffusion_3_5 run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-stable_diffusion_3_5 run fail" >>"${log_dir}/infer_res.log"
fi
echo "*******infer text_to_image_generation-stable_diffusion_3_5 end***********"



echo "*****************pip list********************"
pip list | grep paddle

echo "*****************pip list fastdeploy********************"

pip list | grep fastdeploy

python -c "
import paddle

print('Paddle Commit:', paddle.version.commit)
"

echo "*****************infer result********************"
cat ${log_dir}/infer_res.log

echo exit_code:${exit_code}
exit ${exit_code}
