#!/bin/bash

pip install visualdl
# cd examples/dreambooth
pip install -r requirements_sd3.txt
pip install huggingface_hub==0.23.0
#bash change_paddlenlp_version.sh


wget https://paddlenlp.bj.bcebos.com/models/community/westfish/develop-sdxl/dog.zip
unzip -o dog.zip
rm -rf dog.zip
