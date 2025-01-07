import paddle
import numpy as np
from paddlenlp.transformers import OPTModel, GPTTokenizer

def LayerCase():
    """模型库中间态"""
    model = OPTModel.from_pretrained('facebook/opt-125m')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 11), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 11), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = GPTTokenizer.from_pretrained('facebook/opt-125m')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLimage.pngP!", return_token_type_ids=False)
    inputs = tuple(paddle.to_tensor(v, stop_gradient=False) for (k, v) in inputs_dict.items())
    return inputs


def create_numpy_inputs():
    tokenizer = GPTTokenizer.from_pretrained('facebook/opt-125m')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLimage.pngP!", return_token_type_ids=False)
    inputs = tuple(np.array(v) for (k, v) in inputs_dict.items())
    return inputs
