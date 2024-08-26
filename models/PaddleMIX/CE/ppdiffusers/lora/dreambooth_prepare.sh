#!/bin/bash

pip install -U ppdiffusers visualdl
bash change_paddlenlp_version.sh

# 准备数据
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/dogs.tar.gz
tar -zxvf dogs.tar.gz
rm -rf dogs.tar.gz
