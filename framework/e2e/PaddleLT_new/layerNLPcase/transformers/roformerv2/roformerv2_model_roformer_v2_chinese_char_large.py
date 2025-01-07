import paddle
import numpy as np
from paddlenlp.transformers import RoFormerv2Model, RoFormerv2Tokenizer

def LayerCase():
    """模型库中间态"""
    model = RoFormerv2Model.from_pretrained('roformer_v2_chinese_char_large')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 15), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 15), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = RoFormerv2Tokenizer.from_pretrained('roformer_v2_chinese_char_large')
    inputs_dict = tokenizer("欢迎使用百度飞桨!")
    inputs = tuple(paddle.to_tensor([v], stop_gradient=False) for (k, v) in inputs_dict.items())
    return inputs


def create_numpy_inputs():
    tokenizer = RoFormerv2Tokenizer.from_pretrained('roformer_v2_chinese_char_large')
    inputs_dict = tokenizer("欢迎使用百度飞桨!")
    inputs = tuple(np.array([v]) for (k, v) in inputs_dict.items())
    return inputs
