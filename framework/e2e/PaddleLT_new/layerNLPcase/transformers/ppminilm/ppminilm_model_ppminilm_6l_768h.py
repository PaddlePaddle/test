import paddle
import numpy as np
from paddlenlp.transformers import PPMiniLMModel, PPMiniLMTokenizer

def LayerCase():
    """模型库中间态"""
    model = PPMiniLMModel.from_pretrained('ppminilm-6l-768h')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 15), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 15), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = PPMiniLMTokenizer.from_pretrained('ppminilm-6l-768h')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    inputs = tuple(paddle.to_tensor([v], stop_gradient=False) for (k, v) in inputs_dict.items())
    return inputs


def create_numpy_inputs():
    tokenizer = PPMiniLMTokenizer.from_pretrained('ppminilm-6l-768h')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    inputs = tuple(np.array([v]) for (k, v) in inputs_dict.items())
    return inputs
