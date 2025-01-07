import paddle
import numpy as np
from paddlenlp.transformers import FunnelModel, FunnelTokenizer

def LayerCase():
    """模型库中间态"""
    model = FunnelModel.from_pretrained('funnel-transformer/small-base')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 15), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 15), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = FunnelTokenizer.from_pretrained('funnel-transformer/small-base')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    inputs = tuple(paddle.to_tensor([v], stop_gradient=False) for (k, v) in inputs_dict.items())
    return inputs


def create_numpy_inputs():
    tokenizer = FunnelTokenizer.from_pretrained('funnel-transformer/small-base')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    inputs = tuple(np.array([v]) for (k, v) in inputs_dict.items())
    return inputs
