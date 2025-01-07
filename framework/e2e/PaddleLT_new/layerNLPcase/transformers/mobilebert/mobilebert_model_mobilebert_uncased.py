import paddle
import numpy as np
from paddlenlp.transformers import MobileBertModel, MobileBertTokenizer

def LayerCase():
    """模型库中间态"""
    model = MobileBertModel.from_pretrained('mobilebert-uncased')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 13), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 13), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    inputs = tuple(paddle.to_tensor([v], stop_gradient=False) for (k, v) in inputs_dict.items())
    return inputs


def create_numpy_inputs():
    tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    inputs = tuple(np.array([v]) for (k, v) in inputs_dict.items())
    return inputs
