#!/bin/bash

work_path2=${root_path}/PaddleMIX/ppdiffusers/
echo ${work_path2}/

cd ${work_path2}
# export http_proxy=${proxy}
# export https_proxy=${proxy}
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
#pip install paddlenlp==3.0.0b0
nlp_path=${root_path}/PaddleMIX/PaddleNLP/
pushd ${nlp_path} || exit
pip install -e .
popd


# unset http_proxy
# unset https_proxy
pip install pytest safetensors ftfy fastcore opencv-python einops parameterized requests-mock
