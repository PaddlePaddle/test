rm -rf PaddleNLP
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
pip install -e .
cd csrc
python setup_cuda.py install https://github.com/PaddlePaddle/PaddleNLP/blob/develop/csrc/README.md