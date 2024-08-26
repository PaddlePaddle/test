#!/bin/bash

pip install -U decord
bash change_paddlenlp_version.sh


wget https://paddlenlp.bj.bcebos.com/models/community/westfish/lvdm_datasets/sky_timelapse_lvdm.zip
unzip -o sky_timelapse_lvdm.zip
rm -rf sky_timelapse_lvdm.zip
