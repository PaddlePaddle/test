#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/
echo ${work_path}

log_dir=${root_path}/log_application

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
exit_code=0

# 下载依赖、nltk_data和数据
bash prepare.sh

cd ${work_path}
export FLAGS_use_cuda_managed_memory=true
export FLAGS_allocator_strategy=auto_growth
export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
echo "*******application vision_language_chat begin***********"
(python vision_language_chat.py) 2>&1 | tee ${log_dir}/vision_language_chat.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application vision_language_chat run success" >>"${log_dir}/ce_res.log"
else
    echo "application vision_language_chat run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application vision_language_chat end***********"

echo "*******application grounded_sam begin***********"
(python applications/CVinW/grounded_sam.py) 2>&1 | tee ${log_dir}/grounded_sam.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application grounded_sam run success" >>"${log_dir}/ce_res.log"
else
    echo "application grounded_sam run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application grounded_sam end***********"

echo "*******application automatic_label begin***********"
(python applications/Automatic_label/automatic_label.py) 2>&1 | tee ${log_dir}/automatic_label.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application automatic_label run success" >>"${log_dir}/ce_res.log"
else
    echo "application automatic_label run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application automatic_label end***********"

echo "*******application grounded_sam_inpainting begin***********"
(python grounded_sam_inpainting.py) 2>&1 | tee ${log_dir}/grounded_sam_inpainting.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application grounded_sam_inpainting run success" >>"${log_dir}/ce_res.log"
else
    echo "application grounded_sam_inpainting run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application grounded_sam_inpainting end***********"

echo "*******application grounded_sam_chatglm_qa begin***********"
(python grounded_sam_chatglm.py) 2>&1 | tee ${log_dir}/grounded_sam_chatglm_qa.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application grounded_sam_chatglm_qa run success" >>"${log_dir}/ce_res.log"
else
    echo "application grounded_sam_chatglm_qa run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application grounded_sam_chatglm_qa end***********"

echo "*******application grounded_sam_chatglm begin***********"
(python grounded_sam_chatglm.py) 2>&1 | tee ${log_dir}/grounded_sam_chatglm.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application grounded_sam_chatglm run success" >>"${log_dir}/ce_res.log"
else
    echo "application grounded_sam_chatglm run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application grounded_sam_chatglm end***********"

echo "*******application text_guided_image_inpainting begin***********"
(python text_guided_image_inpainting.py) 2>&1 | tee ${log_dir}/text_guided_image_inpainting.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application text_guided_image_inpainting run success" >>"${log_dir}/ce_res.log"
else
    echo "application text_guided_image_inpainting run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application text_guided_image_inpainting end***********"

echo "*******application text_to_image_generation begin***********"
(python text_to_image_generation.py) 2>&1 | tee ${log_dir}/text_to_image_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application text_to_image_generation run success" >>"${log_dir}/ce_res.log"
else
    echo "application text_to_image_generation run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application text_to_image_generation end***********"

echo "*******application text_guided_image_upscaling begin***********"
(python text_guided_image_upscaling.py) 2>&1 | tee ${log_dir}/text_guided_image_upscaling.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application text_guided_image_upscaling run success" >>"${log_dir}/ce_res.log"
else
    echo "application text_guided_image_upscaling run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application text_guided_image_upscaling end***********"

echo "*******application dual_text_image_guided_generation begin***********"
(python dual_text_image_guided_generation.py) | tee ${log_dir}/dual_text_image_guided_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application dual_text_image_guided_generation run success" >> "${log_dir}/ce_res.log"
else
    echo "application dual_text_image_guided_generation run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******application dual_text_image_guided_generation end***********"

echo "*******application image2image_text_guided_generation begin***********"
(python image2image_text_guided_generation.py) 2>&1 | tee ${log_dir}/image2image_text_guided_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application image2image_text_guided_generation run success" >>"${log_dir}/ce_res.log"
else
    echo "application image2image_text_guided_generation run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application image2image_text_guided_generation end***********"

echo "*******application text2video_generation begin***********"
(python text2video_generation.py) 2>&1 | tee ${log_dir}/text2video_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application text2video_generation run success" >>"${log_dir}/ce_res.log"
else
    echo "application text2video_generation run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application text2video_generation end***********"

export http_proxy=${proxy}
export https_proxy=${proxy}
echo "*******application audio2caption_generation begin***********"
(python audio2caption_generation.py) 2>&1 | tee ${log_dir}/audio2caption_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application audio2caption_generation run success" >>"${log_dir}/ce_res.log"
else
    echo "application audio2caption_generation run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application audio2caption_generation end***********"
unset http_proxy
unset https_proxy

echo "*******application audio2chat_generation begin***********"
(python audio2chat_generation.py) 2>&1 | tee ${log_dir}/audio2chat_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application audio2chat_generation run success" >>"${log_dir}/ce_res.log"
else
    echo "application audio2chat_generation run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application audio2chat_generation end***********"

echo "*******application music_generation begin***********"
(python music_generation.py) 2>&1 | tee ${log_dir}/music_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application music_generation run success" >>"${log_dir}/ce_res.log"
else
    echo "application music_generation run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application music_generation end***********"

echo "*******application audio2img begin***********"
(bash audio2img.sh) 2>&1 | tee ${log_dir}/audio2img.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application audio2img run success" >>"${log_dir}/ce_res.log"
else
    echo "application audio2img run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application audio2img end***********"

echo "*******application audio_text2img begin***********"
(bash audio_text2img.sh) 2>&1 | tee ${log_dir}/audio_text2img.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application audio_text2img run success" >>"${log_dir}/ce_res.log"
else
    echo "application audio_text2img run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application audio_text2img end***********"

echo "*******application auodio_image2image begin***********"
(bash auodio_image2image.sh) 2>&1 | tee ${log_dir}/auodio_image2image.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application auodio_image2image run success" >>"${log_dir}/ce_res.log"
else
    echo "application auodio_image2image run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application auodio_image2image end***********"

echo "*******application script_grounded_sam_chatglm begin***********"
(python applications/Inpainting/grounded_sam_chatglm.py) 2>&1 | tee ${log_dir}/script_grounded_sam_chatglm.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application script_grounded_sam_chatglm run success" >>"${log_dir}/ce_res.log"
else
    echo "application script_grounded_sam_chatglm run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application script_grounded_sam_chatglm end***********"

echo "*******application script_grounded_sam_inpainting begin***********"
(python applications/Inpainting/grounded_sam_inpainting.py) 2>&1 | tee ${log_dir}/script_grounded_sam_inpainting.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application script_grounded_sam_inpainting run success" >>"${log_dir}/ce_res.log"
else
    echo "application script_grounded_sam_inpainting run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application script_grounded_sam_inpainting end***********"

echo "*******application gradio_autolable begin***********"
(python applications/gradio_autolable.py) 2>&1 | tee ${log_dir}/gradio_autolable.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application gradio_autolable run success" >>"${log_dir}/ce_res.log"
else
    echo "application gradio_autolable run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application gradio_autolable end***********"

echo "*******application gradio_text2image begin***********"
(python applications/gradio_text2image.py) 2>&1 | tee ${log_dir}/gradio_text2image.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "application gradio_text2image run success" >>"${log_dir}/ce_res.log"
else
    echo "application gradio_text2image run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******application gradio_text2image end***********"

unset FLAGS_use_cuda_managed_memory
unset FLAGS_allocator_strategy
unset FLAGS_embedding_deterministic
unset FLAGS_cudnn_deterministic
cat ${log_dir}/ce_res.log

echo exit_code:${exit_code}
exit $exit_code
