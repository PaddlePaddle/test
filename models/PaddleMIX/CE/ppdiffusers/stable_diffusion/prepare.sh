#!/bin/bash

echo "*******prepare begin***********"
# export http_proxy=${proxy}
# export https_proxy=${proxy}
pip install -r requirements.txt
bash ${root_path}/PaddleMIX/change_paddlenlp_version.sh


# unset http_proxy
# unset https_proxy

rm -rf data/
# 下载数据集
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/laion400m_demo_data.tar.gz
tar -xf laion400m_demo_data.tar.gz
rm -rf laion400m_demo_data.tar.gz

# 下载权重
wget https://bj.bcebos.com/paddlenlp/models/community/CompVis/CompVis-stable-diffusion-v1-4-paddle-init-pd.tar.gz
# 解压
tar -xf CompVis-stable-diffusion-v1-4-paddle-init-pd.tar.gz
rm -rf CompVis-stable-diffusion-v1-4-paddle-init-pd.tar.gz

echo "*******prepare end***********"
