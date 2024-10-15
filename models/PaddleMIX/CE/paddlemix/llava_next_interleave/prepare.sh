mix_path=${root_path}/PaddeMIX
cd ${mix_path}
pip install -r requirements.txt
pip install -e .

cd ppdiffusers
pip install -r requirements.txt
pip install -e .

cd ..
bash change_paddlenlp_version.sh
