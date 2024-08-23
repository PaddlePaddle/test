#!/bin/bash

# 安装依赖(此模型级别的)
pip install -r requirements.txt
#pip install paddlenlp==3.0.0b0

# Open-Sora训练样本数据下载
wget https://bj.bcebos.com/paddlenlp/models/community/tsaiyue/OpenSoraData/OpenSoraData.tar.gz

# 文件解压
tar -xzvf OpenSoraData.tar.gz
