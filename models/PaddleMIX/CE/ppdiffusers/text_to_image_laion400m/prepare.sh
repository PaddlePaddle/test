#!/bin/bash

# export http_proxy=${proxy}
# export https_proxy=${proxy}
pip install -r requirements.txt
bash change_paddlenlp_version.sh


# unset http_proxy
# unset https_proxy

rm -rf data/
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/laion400m_demo_data.tar.gz
tar -zxvf laion400m_demo_data.tar.gz
rm -rf laion400m_demo_data.tar.gz
