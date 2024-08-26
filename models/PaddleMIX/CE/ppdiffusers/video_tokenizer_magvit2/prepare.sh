#!/bin/bash

cd ${root_path}/PaddleMIX/ppdiffusers/examples/stable_diffusion
pip install beartype
pip install -r requirements.txt
bash ${root_path}/PaddleMIX/change_paddlenlp_version.sh

