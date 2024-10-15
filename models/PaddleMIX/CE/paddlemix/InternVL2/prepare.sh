pip install opencv-python
pip install soundfile
pip install decord

pip install -r requirements.txt
pip install -e .

cd ppdiffusers
pip install -r requirements.txt
pip install -e .

cd ..

bash change_paddlenlp_version.sh


# 数据集下载
cd ${work_path}
mkdir playground
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground/data.tar
tar -xvf data.tar -C playground
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground/opensource.tar
tar -xvf opensource.tar -C playground

rm -rf data.tar opensource.tar




