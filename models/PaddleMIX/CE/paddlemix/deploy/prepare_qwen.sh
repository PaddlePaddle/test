wget https://github.com/PaddlePaddle/PaddleNLP/archive/refs/tags/v2.7.2.tar.gz
rm -rf PaddleNLP
tar xf v2.7.2.tar.gz
mv PaddleNLP-2.7.2/ PaddleNLP/
cd PaddleNLP
pip install -e .
cd csrc
python setup_cuda.py install