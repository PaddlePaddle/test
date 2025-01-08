#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/class_conditional_image_generation/DiT/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0

# 下载依赖和数据
# bash prepare_new.sh
bash prepare.sh

echo "*******class_conditional_image_generation/DiT train begin***********"
(sh 0_run_train_dit_trainer.sh) 2>&1 | tee ${log_dir}/class_conditional_image_generation_DiT_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "class_conditional_image_generation/DiT train run success" >>"${log_dir}/ce_res.log"
else
    echo "class_conditional_image_generation/DiT train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******class_conditional_image_generation/DiT train end***********"


echo "*******class_conditional_image_generation/DiT multi_train begin***********"
(sh 1_run_train_dit_notrainer.sh) 2>&1 | tee ${log_dir}/class_conditional_image_generation_DiT_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "class_conditional_image_generation/DiT multi train run success" >>"${log_dir}/ce_res.log"
else
    echo "class_conditional_image_generation/DiT multi train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******class_conditional_image_generation/DiT multi train end***********"

echo "*******class_conditional_image_generation/DiT auto_train begin***********"
(sh 0_run_train_dit_trainer_auto.sh) 2>&1 | tee ${log_dir}/class_conditional_image_generation_DiT_auto_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "class_conditional_image_generation/DiT auto train run success" >>"${log_dir}/ce_res.log"
else
    echo "class_conditional_image_generation/DiT auto train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******class_conditional_image_generation/DiT auto train end***********"


echo "*******class_conditional_image_generation/DiT large_train begin***********"
(sh 4_run_train_largedit_3b_trainer_auto.sh) 2>&1 | tee ${log_dir}/class_conditional_image_generation_DiT_large_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "class_conditional_image_generation/DiT large train run success" >>"${log_dir}/ce_res.log"
else
    echo "class_conditional_image_generation/DiT large train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******class_conditional_image_generation/DiT large train end***********"



echo "*******class_conditional_image_generation/DiT infer begin***********"
(python infer_demo_dit.py) 2>&1 | tee ${log_dir}/class_conditional_image_generation_DiT_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "class_conditional_image_generation/DiT infer run success" >>"${log_dir}/ce_res.log"
else
    echo "class_conditional_image_generation/DiT infer run fail" >>"${log_dir}/ce_res.log"
fi

echo "*******class_conditional_image_generation/DiT transfer begin***********"
(python tools/convert_dit_to_ppdiffusers.py \
    --image_size 256 \
    --model_name DiT_XL_2 \
    --model_weights output_trainer/DiT_XL_patch2_trainer/model_state.pdparams \
    --checkpoint_path output_trainer/DiT_XL_patch2_trainer/checkpoint-10/) 2>&1 | tee ${log_dir}/class_conditional_image_generation_DiT_transfer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "class_conditional_image_generation/DiT transfer run success" >>"${log_dir}/ce_res.log"
else
    echo "class_conditional_image_generation/DiT transfer run fail" >>"${log_dir}/ce_res.log"
fi


echo "*******class_conditional_image_generation/DiT infer_generate begin***********"
(python generate_infer_test.py) 2>&1 | tee ${log_dir}/class_conditional_image_generation_DiT_infer_generate.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "class_conditional_image_generation/DiT infer_generate run success" >>"${log_dir}/ce_res.log"
else
    echo "class_conditional_image_generation/DiT infer_generate run fail" >>"${log_dir}/ce_res.log"
fi



echo exit_code:${exit_code}
exit ${exit_code}
