wget https://github.com/PaddlePaddle/PaddleNLP/archive/refs/tags/v2.7.2.tar.gz
tar xf v2.7.2.tar.gz
cd PaddleNLP
pip install -e .
cd csrc
python setup_cuda.py install