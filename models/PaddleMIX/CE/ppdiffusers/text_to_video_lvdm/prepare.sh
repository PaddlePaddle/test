#!/bin/bash

pip install -U decord
#pip install paddlenlp==3.0.0b0
nlp_path=${root_path}/PaddleMIX/PaddleNLP/
pushd ${nlp_path} || exit
cd ${nlp_path}
pip install -e .
popd

wget https://paddlenlp.bj.bcebos.com/models/community/westfish/lvdm_datasets/sky_timelapse_lvdm.zip
unzip -o sky_timelapse_lvdm.zip
rm -rf sky_timelapse_lvdm.zip
