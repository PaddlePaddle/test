#!/bin/bash
#pip install paddlenlp==3.0.0b0

rm -rf data
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/laion400m_demo_data.tar.gz
tar -zxvf laion400m_demo_data.tar.gz
rm -rf laion400m_demo_data.tar.gz
