#!/bin/bash
pip install pytest safetensors ftfy fastcore opencv-python einops parameterized requests-mock

work_path2=${root_path}/PaddleMIX/ppdiffusers/
echo ${work_path2}/

cd ${work_path2}

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .


