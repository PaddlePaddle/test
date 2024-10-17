#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/text_to_image/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}
/bin/cp -rf ../change_paddlenlp_version.sh ${work_path}

exit_code=0

cd ${work_path}
bash prepare.sh

export FLAGS_npu_storage_format=0
export FLAGS_use_stride_kernel=0


echo "*******paddlemix text_to_image_sdxl_train begin begin***********"
(bash train.sh) 2>&1 | tee ${log_dir}/text_to_image_sdxl_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image_sdxl_train run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image_sdxl_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix text_to_image_sdxl_train end***********"


echo "*******paddlemix text_to_image_sdxl_infer begin begin***********"
(python test_infer.py) 2>&1 | tee ${log_dir}/text_to_image_sdxl_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image_sdxl_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image_sdxl_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix text_to_image_sdxl_infer end***********"

echo "*******paddlemix text_to_image_sdxl_multi_infer begin begin***********"
(python test_multi_checkpoint_infer.py) 2>&1 | tee ${log_dir}/text_to_image_sdxl_multi_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image_sdxl_multi_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image_sdxl_multi_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix text_to_image_sdxl_multi_infer end***********"


# export FLAGS_npu_storage_format=0
# export FLAGS_use_stride_kernel=0
# export FLAGS_npu_scale_aclnn=True
# export FLAGS_allocator_strategy=auto_growth


# echo "*******paddlemix text_to_image_sdxl_lora_train begin begin***********"
# (bash lora_train.sh) 2>&1 | tee ${log_dir}/text_to_image_sdxl_lora_train.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "text_to_image_sdxl_lora_train run success" >>"${log_dir}/ce_res.log"
# else
#     echo "text_to_image_sdxl_lora_train run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******paddlemix text_to_image_sdxl_lora_train end***********"


# echo "*******paddlemix text_to_image_sdxl_lora_train_unet begin begin***********"
# (bash lora_train_unet.sh) 2>&1 | tee ${log_dir}/text_to_image_sdxl_lora_train_unet.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "text_to_image_sdxl_lora_train_unet run success" >>"${log_dir}/ce_res.log"
# else
#     echo "text_to_image_sdxl_lora_train_unet run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******paddlemix text_to_image_sdxl_lora_train_unet end***********"

# echo "*******paddlemix text_to_image_sdxl_lora_infer begin begin***********"
# (python test_lora_infer.py) 2>&1 | tee ${log_dir}/text_to_image_sdxl_lora_infer.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "text_to_image_sdxl_lora_infer run success" >>"${log_dir}/ce_res.log"
# else
#     echo "text_to_image_sdxl_lora_infer run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******paddlemix text_to_image_sdxl_lora_infer end***********"

# echo "*******paddlemix text_to_image_sdxl_multi_lora_infer begin begin***********"
# (python test_lora_multi_infer.py) 2>&1 | tee ${log_dir}/text_to_image_sdxl_multi_lora_infer.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "text_to_image_sdxl_multi_lora_infer run success" >>"${log_dir}/ce_res.log"
# else
#     echo "text_to_image_sdxl_multi_lora_infer run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******paddlemix text_to_image_sdxl_multi_lora_infer end***********"


unset http_proxy
unset https_proxy



# 查看结果
cat ${log_dir}/ce_res.log

echo exit_code:${exit_code}
exit ${exit_code}
