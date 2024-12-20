#!/bin/bash
pip install pytest safetensors ftfy fastcore opencv-python einops parameterized requests-mock
pip install pytest safetensors ftfy fastcore opencv-python einops parameterized requests-mock
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
pip install pytest-xdist
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers diffusers

work_path2=${root_path}/PaddleMIX/ppdiffusers/
echo ${work_path2}/

cd ${work_path2}

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

pip uninstall paddlenlp -y
git clone https://github.com/PaddlePaddle/PaddleNLP
cd PaddleNLP
pip install -e .

pip list | grep paddle