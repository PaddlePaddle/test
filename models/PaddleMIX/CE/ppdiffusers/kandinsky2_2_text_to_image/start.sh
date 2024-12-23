#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/kandinsky2_2/text_to_image/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0


# echo "*******kandinsky2_2_text_to_image finetune_decoder begin***********"
# (bash finetune_decoder.sh) 2>&1 | tee ${log_dir}/kandinsky2_2_text_to_image_finetune_decoder.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "kandinsky2_2_text_to_image finetune_decoder run success" >>"${log_dir}/ce_res.log"
# else
#     echo "kandinsky2_2_text_to_image finetune_decoder run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******kandinsky2_2_text_to_image finetune_decoder end***********"


# echo "******kandinsky2_2_text_to_image infer_decoder begin***********"
# (python infer_decoder.py 2>&1) | tee ${log_dir}/kandinsky2_2_text_to_image_infer_decoder.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "kandinsky2_2_text_to_image infer_decoder run success" >>"${log_dir}/ce_res.log"
# else
#     echo "kandinsky2_2_text_to_image infer_decoder run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******kandinsky2_2_text_to_image infer_decoder end***********"


# echo "*******kandinsky2_2_text_to_image finetune_prior begin***********"
# (bash finetune_prior.sh) 2>&1 | tee ${log_dir}/kandinsky2_2_text_to_image_finetune_prior.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "kandinsky2_2_text_to_image finetune_prior run success" >>"${log_dir}/ce_res.log"
# else
#     echo "kandinsky2_2_text_to_image finetune_prior run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******kandinsky2_2_text_to_image finetune_prior end***********"


# echo "*******kandinsky2_2_text_to_image infer_prior begin***********"
# (python infer_prior.py) 2>&1 | tee ${log_dir}/kandinsky2_2_text_to_image_infer_prior.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "kandinsky2_2_text_to_image infer_prior run success" >>"${log_dir}/ce_res.log"
# else
#     echo "kandinsky2_2_text_to_image infer_prior run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******kandinsky2_2_text_to_image infer_prior end***********"


# echo "*******kandinsky2_2_text_to_image decoder_multi_train begin***********"
# (bash decoder_multi_train.sh) 2>&1 | tee ${log_dir}/kandinsky2_2_text_to_image_decoder_multi_trainlog
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "kandinsky2_2_text_to_image decoder_multi_train run success" >>"${log_dir}/ce_res.log"
# else
#     echo "kandinsky2_2_text_to_image decoder_multi_train run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******kandinsky2_2_text_to_image decoder_multi_train end***********"


echo "*******kandinsky2_2_text_to_image lora_finetune_decoder begin***********"
(bash lora_finetune_decoder.sh) 2>&1 | tee ${log_dir}/kandinsky2_2_text_to_image_lora_finetune_decoder.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "kandinsky2_2_text_to_image lora_finetune_decoder run success" >>"${log_dir}/ce_res.log"
else
    echo "kandinsky2_2_text_to_image lora_finetune_decoder run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******kandinsky2_2_text_to_image lora_finetune_decoder end***********"


echo "******kandinsky2_2_text_to_image lora_decoder_infer begin***********"
(python lora_decoder_infer.py 2>&1) | tee ${log_dir}/kandinsky2_2_text_to_image_lora_decoder_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "kandinsky2_2_text_to_image lora_decoder_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "kandinsky2_2_text_to_image lora_decoder_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******kandinsky2_2_text_to_image lora_decoder_infer end***********"


echo "*******kandinsky2_2_text_to_image lora_finetune_prior begin***********"
(bash lora_finetune_prior.sh) 2>&1 | tee ${log_dir}/kandinsky2_2_text_to_image_lora_finetune_prior.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "kandinsky2_2_text_to_image lora_finetune_prior run success" >>"${log_dir}/ce_res.log"
else
    echo "kandinsky2_2_text_to_image lora_finetune_prior run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******kandinsky2_2_text_to_image lora_finetune_prior end***********"


echo "******kandinsky2_2_text_to_image lora_prior_infer begin***********"
(python lora_prior_infer.py 2>&1) | tee ${log_dir}/kandinsky2_2_text_to_image_lora_prior_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "kandinsky2_2_text_to_image lora_prior_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "kandinsky2_2_text_to_image lora_prior_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******kandinsky2_2_text_to_image lora_prior_infer end***********"

# # 查看结果
# cat ${log_dir}/ce_res.log
rm -rf ${work_path}/kandi2-decoder-pokemon-model/*
rm -rf ${work_path}/robot-pokemon_infer_decoder.png
rm -rf ${work_path}/kandi2-prior-pokemon-model/*
rm -rf ${work_path}/robot-pokemon_infer_prior.png
rm -rf ${work_path}/kandi22-decoder-pokemon-lora/*
rm -rf ${work_path}/robot_pokemon_lora_decoder_infer.png
rm -rf ${work_path}/kandi22-prior-pokemon-lora/*
rm -rf ${work_path}/robot_pokemon_lora_prior_infer.png

echo exit_code:${exit_code}
exit ${exit_code}
