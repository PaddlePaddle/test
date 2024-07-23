wget https://codeload.github.com/PaddlePaddle/PaddleNLP/zip/refs/tags/v2.7.2
tar xf v2.7.2
cd PaddleNLP
pip install -e .
cd csrc
python setup_cuda.py install