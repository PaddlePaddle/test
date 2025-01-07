import paddle
import numpy as np
from paddlenlp.transformers import UNIMOModel, UNIMOTokenizer

def LayerCase():
    """模型库中间态"""
    model = UNIMOModel.from_pretrained('unimo-text-1.0-lcsts-new')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 15), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 15), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = UNIMOTokenizer.from_pretrained('unimo-text-1.0-lcsts-new')
    inputs_dict = tokenizer.gen_encode("Welcome to use PaddlePaddle and PaddleNLP!", return_tensors=True)
    inputs = tuple(paddle.to_tensor(v, stop_gradient=False) for (k, v) in inputs_dict.items())
    return inputs


def create_numpy_inputs():
    tokenizer = UNIMOTokenizer.from_pretrained('unimo-text-1.0-lcsts-new')
    inputs_dict = tokenizer.gen_encode("Welcome to use PaddlePaddle and PaddleNLP!", return_tensors=True)
    inputs = tuple(np.array(v) for (k, v) in inputs_dict.items())
    return inputs
