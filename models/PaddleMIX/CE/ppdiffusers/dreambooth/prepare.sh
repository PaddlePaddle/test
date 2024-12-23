#!/bin/bash
pip install visualdl
#bash ${root_path}/PaddleMIX/change_paddlenlp_version.sh
pip install huggingface_hub==0.23.0

wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/dogs.tar.gz
tar -xf dogs.tar.gz
rm -rf dogs.tar.gz
