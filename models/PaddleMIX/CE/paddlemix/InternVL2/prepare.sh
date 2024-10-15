cp ../change_paddlenlp_version.sh ${work_path}

pip install cv2
pip install -r requirements.txt
pip install -e .

cd ppdiffusers
pip install -r requirements.txt
pip install -e .

cd ..

bash change_paddlenlp_version.sh





