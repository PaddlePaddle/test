import paddle
import numpy as np
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

def LayerCase():
    """模型库中间态"""
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 14), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 14), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 14), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    inputs = tuple(paddle.to_tensor([v], stop_gradient=False) for (k, v) in inputs_dict.items())
    return inputs


def create_numpy_inputs():
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    inputs = tuple(np.array([v]) for (k, v) in inputs_dict.items())
    return inputs
