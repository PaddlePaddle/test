import paddle
import numpy as np
from paddlenlp.transformers import ProphetNetModel, ProphetNetTokenizer

def LayerCase():
    """模型库中间态"""
    model = ProphetNetModel.from_pretrained('prophetnet-large-uncased')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 12), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 12), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = ProphetNetTokenizer.from_pretrained('prophetnet-large-uncased')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", return_position_ids=True) # 随便构建的输入, token的key实际上和模型输入参数没对上
    inputs = tuple(paddle.to_tensor([v], stop_gradient=False) for (k, v) in inputs_dict.items())
    return inputs


def create_numpy_inputs():
    tokenizer = ProphetNetTokenizer.from_pretrained('prophetnet-large-uncased')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", return_position_ids=True) # 随便构建的输入, token的key实际上和模型输入参数没对上
    inputs = tuple(np.array([v]) for (k, v) in inputs_dict.items())
    return inputs
