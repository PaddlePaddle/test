import paddle
import numpy as np
from paddlenlp.transformers import NeZhaModel, NeZhaTokenizer

def LayerCase():
    """模型库中间态"""
    model = NeZhaModel.from_pretrained('nezha-base-wwm-chinese')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 11), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 11), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = NeZhaTokenizer.from_pretrained('nezha-base-wwm-chinese')
    inputs_dict = tokenizer("欢迎使用百度飞浆!", return_tensors='pd')
    inputs = tuple(paddle.to_tensor(v, stop_gradient=False) for (k, v) in inputs_dict.items())
    return inputs


def create_numpy_inputs():
    tokenizer = NeZhaTokenizer.from_pretrained('nezha-base-wwm-chinese')
    inputs_dict = tokenizer("欢迎使用百度飞浆!", return_tensors='pd')
    inputs = tuple(np.array(v) for (k, v) in inputs_dict.items())
    return inputs
