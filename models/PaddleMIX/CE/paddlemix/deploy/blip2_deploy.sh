#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/deploy/blip2/
echo ${work_path}

log_dir=${root_path}/log_deploy

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
exit_code=0

export http_proxy=${proxy}
export https_proxy=${proxy}
git clone https://github.com/PaddlePaddle/PaddleNLP.git -b develop
cd PaddleNLP
pip install -e .
cd csrc
python setup_cuda.py install
unset http_proxy
unset https_proxy

cd ${work_path}

echo "*******paddlemix deploy blip2 begin***********"

bash prepare.sh

# python export_model.py --model_name_or_path /root/.paddlenlp/models/facebook/opt-2.7b --output_path opt-2.7b-export --dtype float16 --inference_model --model_prefix=opt --model_type=opt-img2txt

#visual encoder 和 Qformer 静态图模型导出
(python export.py \
    --model_name_or_path paddlemix/blip2-caption-opt2.7b) 2>&1 | tee ${log_dir}/run_deploy_blip2_export.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix deploy blip2 export run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix deploy blip2 export run fail" >>"${log_dir}/ce_res.log"
fi

#静态图预测
(python predict.py \
    --first_model_path blip2_export/image_encoder \
    --second_model_path opt-2.7b-infer_static/opt \
    --image_path https://paddlenlp.bj.bcebos.com/data/images/mugs.png \
    --prompt "a photo of") 2>&1 | tee ${log_dir}/run_deploy_blip2_predict.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix deploy blip2 predict run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix deploy blip2 predict run fail" >>"${log_dir}/ce_res.log"
fi

echo "*******paddlemix deploy blip2 end***********"

# 检查命令是否成功执行
if [ ${exit_code} -ne 0 ]; then
    exit 1
fi
