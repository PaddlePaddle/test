#!/bin/bash

wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/fill50k.zip
unzip -o fill50k.zip
rm -rf fill50k.zip

pip install -r requirements.txt
pip install gradio
pip install huggingface_hub==0.23.0
#bash ${root_path}/PaddleMIX/change_paddlenlp_version.sh


export http_proxy=${proxy}
export https_proxy=${proxy}
wget https://user-images.githubusercontent.com/50394665/221844474-fd539851-7649-470e-bded-4d174271cc7f.png
unset http_proxy
unset https_proxy
