#!/bin/bash

pip install -r requirements.txt
#pip install paddlenlp==3.0.0b0
nlp_path=${root_path}/PaddleMIX/PaddleNLP/
pushd ${nlp_path} || exit
pip install -e .
popd

export http_proxy=${mix_proxy}
export https_proxy=${mix_proxy}
pip install git+https://github.com/arogozhnikov/einops.git
unset http_proxy
unset https_proxy

mkdir -p datasets/coco256_features

wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/coco256_features/run_vis.tar
tar -xvf run_vis.tar -C ./datasets/coco256_features
rm -rf run_vis.tar

wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/coco256_features/empty_context.tar
tar -xvf empty_context.tar -C ./datasets/coco256_features
rm -rf empty_context.tar

wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/coco256_features/val.tar
tar -xvf val.tar -C ./datasets/coco256_features
rm -rf val.tar

wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/coco256_features/train.tar
tar -xvf train.tar -C ./datasets/coco256_features
rm -rf train.tar
