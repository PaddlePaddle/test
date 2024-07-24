#!/bin/bash

# pip uninstall -y paddlenlp
# wget http://10.99.15.135:8000/paddlenlp-2.6.2-py3-none-any.whl
# pip install paddlenlp-2.6.2-py3-none-any.whl
# rm -rf paddlenlp-2.6.2-py3-none-any.whl
pip install paddlenlp==2.7.2

work_path=${root_path}/PaddleMIX/ppdiffusers/
echo ${work_path}/

cd ${work_path}
pip install -e .