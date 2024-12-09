git submodule update --init --recursive
cd PaddleNLP
git reset --hard 498f70988431be278dac618411fbfb0287853cd9
pip install -e .
cd csrc
python setup_cuda.py install