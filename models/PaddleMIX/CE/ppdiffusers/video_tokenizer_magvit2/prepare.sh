#!/bin/bash

cd ${root_path}/PaddleMIX/ppdiffusers/examples/stable_diffusion
pip install beartype
pip install -r requirements.txt
#pip install paddlenlp==3.0.0b0
nlp_path=${root_path}/PaddleMIX/PaddleNLP/
pushd ${nlp_path} || exit
pip install -e .
popd
