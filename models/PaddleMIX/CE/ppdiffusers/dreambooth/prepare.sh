#!/bin/bash
#pip install paddlenlp==3.0.0b0
nlp_path=${root_path}/PaddleMIX/PaddleNLP/
pushd ${nlp_path} || exit
pip install -e .
popd

wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/dogs.tar.gz
tar -zxvf dogs.tar.gz
rm -rf dogs.tar.gz
