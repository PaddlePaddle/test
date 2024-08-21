#!/bin/bash

# cd ${root_path}/PaddleMIX/ppdiffusers
# pip install -e .
# pip install -r requirements.txt
pip install pytest safetensors ftfy fastcore opencv-python einops parameterized requests-mock
pip install -U ppdiffusers
pip install imageio
pip install tiktoken

cd ${root_path}/PaddleMIX/
pip install -r requirements.txt
python -m pip install --upgrade pip
pip install -e .
pip install -r paddlemix/appflow/requirements.txt

export http_proxy=${proxy}
export https_proxy=${proxy}
python nltk_data_download.py
unset http_proxy
unset https_proxy

wget https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/zh.wav

 python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
 pip install paddlenlp==3.0.0b1