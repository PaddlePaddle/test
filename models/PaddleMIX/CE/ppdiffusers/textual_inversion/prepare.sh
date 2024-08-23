#!/bin/bash
#pip install paddlenlp==3.0.0b0
nlp_path=${root_path}/PaddleMIX/PaddleNLP/
pushd ${nlp_path} || exit
pip install -e .
popd

wget https://paddle-qa.bj.bcebos.com/PaddleMIX/cat_toy_images.tar.gz
tar -zxvf cat_toy_images.tar.gz
rm -rf cat_toy_images.tar.gz
