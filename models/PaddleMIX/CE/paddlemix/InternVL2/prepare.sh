cp ../change_paddlenlp_version.sh ${work_path}

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





