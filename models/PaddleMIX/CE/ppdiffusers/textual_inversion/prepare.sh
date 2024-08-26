#!/bin/bash
bash change_paddlenlp_version.sh

wget https://paddle-qa.bj.bcebos.com/PaddleMIX/cat_toy_images.tar.gz
tar -zxvf cat_toy_images.tar.gz
rm -rf cat_toy_images.tar.gz
