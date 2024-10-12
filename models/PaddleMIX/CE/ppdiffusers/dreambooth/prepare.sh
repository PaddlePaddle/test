#!/bin/bash
pip install visualdl
bash ${root_path}/PaddleMIX/change_paddlenlp_version.sh


wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/dogs.tar.gz
tar -zxvf dogs.tar.gz
rm -rf dogs.tar.gz
