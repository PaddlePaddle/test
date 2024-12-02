#!/bin/bash

# pip uninstall -y paddlenlp
# wget http://10.99.15.135:8000/paddlenlp-2.6.2-py3-none-any.whl
# pip install paddlenlp-2.6.2-py3-none-any.whl
# rm -rf paddlenlp-2.6.2-py3-none-any.whl
work_path=${root_path}/PaddleMIX/ppdiffusers/
echo ${work_path}/

cd ${work_path}
pip install -e .

cp ../change_paddlenlp_version.sh ${root_path}/PaddleMIX
bash ${root_path}/PaddleMIX/change_paddlenlp_version.sh