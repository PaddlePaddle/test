import paddle
import numpy as np
from paddlenlp.transformers import DalleBartModel, DalleBartTokenizer

def LayerCase():
    """模型库中间态"""
    model = DalleBartModel.from_pretrained('dalle-mini')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 15), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 15), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = DalleBartTokenizer.from_pretrained('dalle-mini')
    inputs_dict = tokenizer("graphite sketch of Elon Musk")
    inputs = tuple(paddle.to_tensor([v], stop_gradient=False) for (k, v) in inputs_dict.items())
    return inputs


def create_numpy_inputs():
    tokenizer = DalleBartTokenizer.from_pretrained('dalle-mini')
    inputs_dict = tokenizer("graphite sketch of Elon Musk")
    inputs = tuple(np.array([v]) for (k, v) in inputs_dict.items())
    return inputs
