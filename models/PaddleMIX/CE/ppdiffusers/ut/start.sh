#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers
echo ${work_path}

log_dir=${root_path}/ut_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}
/bin/cp -f ../change_paddlenlp_version.sh ${work_path}
cd ${work_path}
exit_code=0


pip install pytest safetensors ftfy fastcore opencv-python einops parameterized requests-mock
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
pip install pytest-xdist
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers diffusers
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
bash ${root_path}/PaddleMIX/change_paddlenlp_version.sh

export http_proxy=${mix_proxy}
export https_proxy=${mix_proxy}
# rm -rf tests/pipelines/test_pipelines.py
# rm -rf tests/pipelines/stable_diffusion/test_stable_diffusion_pix2pix_zero.py

exit_code=0

export HF_ENDPOINT=https://hf-mirror.com
export no_proxy=baidu.com,127.0.0.1,0.0.0.0,localhost,bcebos.com,pip.baidu-int.com,mirrors.baidubce.com,repo.baidubce.com,repo.bcm.baidubce.com,pypi.tuna.tsinghua.edu.cn,aistudio.baidu.com
export USE_PPXFORMERS=True
export RUN_SLOW=True

echo "*******ppdiffusers ut tests begin***********"
(python -m pytest -v tests/models) 2>&1 | tee ${log_dir}/tests_ut.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers ut tests  run success" >>"${log_dir}/ut_res.log"
else
    echo "ppdiffusers ut tests  run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******ppdiffusers ut tests end***********"

unset http_proxy
unset https_proxy

# # 查看结果
cat ${log_dir}/ut_res.log

# 查看pip list 
pip list | grep paddle

pip list | grep huggingface
echo exit_code:${exit_code}
exit ${exit_code}
