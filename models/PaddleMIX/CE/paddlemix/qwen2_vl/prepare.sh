mix_path=${root_path}/PaddleMIX
cd ${mix_path}

pip install -r requirements.txt
pip install -e .

cd ppdiffusers
pip install -r requirements.txt
pip install -e .
pip install tiktoken

cd ..

bash change_paddlenlp_version.sh

# 数据集下载
cd ${mix_path}
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground.tar
tar xf playground.tar
cd playground
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground/opensource_json.tar
tar xf opensource_json.tar

