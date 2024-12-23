
pip install -r requirements_svd.txt
pip install huggingface_hub==0.23.0

wget https://paddlenlp.bj.bcebos.com/models/community/westfish/lvdm_datasets/sky_timelapse_lvdm.zip && unzip sky_timelapse_lvdm.zip
wget https://example.com/dataset.zip && unzip dataset.zip
rm -rf dataset.zip
rm -rf sky_timelapse_lvdm.zip