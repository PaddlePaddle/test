
cp ../change_paddlenlp_version.sh ${root_path}/PaddleMIX

cd ${root_path}/PaddleMIX
pip install -r requirements.txt
pip install -e .

cd ppdiffusers
pip install -r requirements.txt
pip install -e .

cd ..

bash change_paddlenlp_version.sh





