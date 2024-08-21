#!/bin/bash

pip install -r requirements.txt
pip install paddlenlp==3.0.0b0


rm -rf fastdit_imagenet256_tiny/
wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/fastdit_features/fastdit_imagenet256_tiny.tar
tar -xvf fastdit_imagenet256_tiny.tar
rm -rf fastdit_imagenet256_tiny.tar

