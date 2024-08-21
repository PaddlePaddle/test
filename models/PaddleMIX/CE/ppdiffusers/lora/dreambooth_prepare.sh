#!/bin/bash

pip install -U ppdiffusers visualdl
pip install paddlenlp==3.0.0b0

# 准备数据
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/dogs.tar.gz
tar -zxvf dogs.tar.gz
rm -rf dogs.tar.gz
