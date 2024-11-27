#!/bin/bash

pip install -r requirements.txt

bash ${root_path}/PaddleMIX/change_paddlenlp_version.sh


rm -rf fastdit_imagenet256_tiny/
wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/fastdit_features/fastdit_imagenet256_tiny.tar
tar -xvf fastdit_imagenet256_tiny.tar
rm -rf fastdit_imagenet256_tiny.tar

