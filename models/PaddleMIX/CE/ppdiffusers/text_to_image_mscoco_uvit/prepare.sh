#!/bin/bash

pip install -r requirements.txt
#bash ${root_path}/PaddleMIX/change_paddlenlp_version.sh


export http_proxy=${mix_proxy}
export https_proxy=${mix_proxy}
pip install git+https://github.com/arogozhnikov/einops.git
unset http_proxy
unset https_proxy

mkdir -p datasets/coco256_features

wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/coco256_features/run_vis.tar
tar -xf run_vis.tar -C ./datasets/coco256_features
rm -rf run_vis.tar

wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/coco256_features/empty_context.tar
tar -xf empty_context.tar -C ./datasets/coco256_features
rm -rf empty_context.tar

wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/coco256_features/val.tar
tar -xf val.tar -C ./datasets/coco256_features
rm -rf val.tar

wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/coco256_features/train.tar
tar -xf train.tar -C ./datasets/coco256_features
rm -rf train.tar
