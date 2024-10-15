#!/bin/bash
cd ${root_path}/PaddleMIX/ppdiffusers/
pip install -e .

pip install visualdl
cd examples/text_to_image
pip install -r requirements_sdxl.txt
bash change_paddlenlp_version.sh