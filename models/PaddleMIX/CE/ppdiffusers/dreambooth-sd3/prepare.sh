#!/bin/bash
cd ${root_path}/ppdiffusers/
pip install -e .

pip install visualdl
cd examples/dreambooth
pip install -r requirements_sd3.txt

bash ${root_path}/PaddleMIX/change_paddlenlp_version.sh


wget https://paddlenlp.bj.bcebos.com/models/community/westfish/develop-sdxl/dog.zip
unzip dog.zip
rm -rf dog.zip
