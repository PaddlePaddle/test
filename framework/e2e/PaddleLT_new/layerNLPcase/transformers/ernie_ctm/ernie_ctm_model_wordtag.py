import paddle
import numpy as np
from paddlenlp.transformers import ErnieCtmModel, ErnieCtmTokenizer

def LayerCase():
    """模型库中间态"""
    model = ErnieCtmModel.from_pretrained('wordtag')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 15), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 15), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = ErnieCtmTokenizer.from_pretrained('wordtag')
    inputs_dict = tokenizer("He was a puppeteer")
    inputs = tuple(paddle.to_tensor([v], stop_gradient=False) for (k, v) in inputs_dict.items())
    return inputs


def create_numpy_inputs():
    tokenizer = ErnieCtmTokenizer.from_pretrained('wordtag')
    inputs_dict = tokenizer("He was a puppeteer")
    inputs = tuple(np.array([v]) for (k, v) in inputs_dict.items())
    return inputs
