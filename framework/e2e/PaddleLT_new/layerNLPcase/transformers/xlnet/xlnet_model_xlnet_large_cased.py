import paddle
import numpy as np
from paddlenlp.transformers.xlnet.modeling import XLNetModel
from paddlenlp.transformers.xlnet.tokenizer import XLNetTokenizer

def LayerCase():
    """模型库中间态"""
    model = XLNetModel.from_pretrained('xlnet-large-cased')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 13), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 13), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    inputs_dict = tokenizer("Hey, Paddle-paddle is awesome !")
    inputs = tuple(paddle.to_tensor([v], stop_gradient=False) for (k, v) in inputs_dict.items())
    return inputs


def create_numpy_inputs():
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    inputs_dict = tokenizer("Hey, Paddle-paddle is awesome !")
    inputs = tuple(np.array([v]) for (k, v) in inputs_dict.items())
    return inputs
