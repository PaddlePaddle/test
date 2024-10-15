mix_path=${root_path}/PaddeMIX
cd ${mix_path}
rm -rf paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl
wget install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Linux-Gpu-Cuda11.8-Cudnn8.6-Mkl-Avx-Gcc8.2-SelfBuiltPypiUse/latest/paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl
python -m pip install paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl --force-reinstall

pip install -r requirements.txt
pip install -e .

cd ppdiffusers
pip install -r requirements.txt
pip install -e .
pip install tiktoken

cd ..

bash change_paddlenlp_version.sh
rm -rf paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl

