#!/bin/bash

#bash ${root_path}/PaddleMIX/change_paddlenlp_version.sh


rm -rf data
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/laion400m_demo_data.tar.gz
tar -xf laion400m_demo_data.tar.gz
rm -rf laion400m_demo_data.tar.gz
