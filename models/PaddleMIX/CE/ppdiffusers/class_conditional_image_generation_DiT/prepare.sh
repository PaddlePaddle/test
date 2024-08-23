#!/bin/bash

pip install -r requirements.txt
#pip install paddlenlp==3.0.0b0
nlp_path=${root_path}/PaddleMIX/PaddleNLP/
pushd ${nlp_path} || exit
pip install -e .
popd

rm -rf fastdit_imagenet256_tiny/
wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/fastdit_features/fastdit_imagenet256_tiny.tar
tar -xvf fastdit_imagenet256_tiny.tar
rm -rf fastdit_imagenet256_tiny.tar

