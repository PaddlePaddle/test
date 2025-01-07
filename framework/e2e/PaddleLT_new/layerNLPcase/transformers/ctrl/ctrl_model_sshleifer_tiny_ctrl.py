import paddle
import numpy as np
from paddlenlp.transformers import CTRLModel, CTRLTokenizer

def LayerCase():
    """模型库中间态"""
    model = CTRLModel.from_pretrained('sshleifer-tiny-ctrl')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 13), dtype=paddle.float32, stop_gradient=False),
        None,
        paddle.static.InputSpec(shape=(-1, 13), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = CTRLTokenizer.from_pretrained('sshleifer-tiny-ctrl')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    inputs = (
        paddle.to_tensor([inputs_dict['input_ids']], stop_gradient=False),
        None,
        paddle.to_tensor([inputs_dict['token_type_ids']], stop_gradient=False),
    )
    # inputs = tuple(paddle.to_tensor([v], stop_gradient=False) for (k, v) in inputs_dict.items())
    return inputs


def create_numpy_inputs():
    tokenizer = CTRLTokenizer.from_pretrained('sshleifer-tiny-ctrl')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    inputs = (
        np.array([inputs_dict['input_ids']]),
        None,
        np.array([inputs_dict['token_type_ids']]),
    )
    # inputs = tuple(np.array([v]) for (k, v) in inputs_dict.items())
    print(inputs)
    return inputs
