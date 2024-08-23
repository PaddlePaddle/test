#!/bin/bash

# export http_proxy=${proxy}
# export https_proxy=${proxy}
pip install -r requirements.txt
#pip install paddlenlp==3.0.0b0
nlp_path=${root_path}/PaddleMIX/PaddleNLP/
pushd ${nlp_path} || exit
pip install -e .
popd

# unset http_proxy
# unset https_proxy

rm -rf data/
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/laion400m_demo_data.tar.gz
tar -zxvf laion400m_demo_data.tar.gz
rm -rf laion400m_demo_data.tar.gz
