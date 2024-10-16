mix_path=${root_path}/PaddeMIX
cd ${mix_path}
pip install -r requirements.txt
pip install -e .

cd ppdiffusers
pip install -r requirements.txt
pip install -e .

cd ..

bash change_paddlenlp_version.sh

work_pat=${root_path}/PaddeMIX/ppdiffusers/examples/text_to_image
pip install -r requirements_sdxl.txt
bash change_paddlenlp_version.sh


