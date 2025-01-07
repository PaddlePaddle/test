import paddle
import numpy as np
from paddlenlp.transformers import T5Model, T5Tokenizer

def LayerCase():
    """模型库中间态"""
    model = T5Model.from_pretrained('t5-large')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 15), dtype=paddle.float32, stop_gradient=False),
        None,
        paddle.static.InputSpec(shape=(-1, 5), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP! ")
    decoder_inputs = tokenizer("It means you can")
    inputs = (
        paddle.to_tensor([inputs_dict["input_ids"]], dtype="int64", stop_gradient=False),
        None,
        paddle.to_tensor([decoder_inputs["input_ids"]], dtype="int64", stop_gradient=False),
    )
    return inputs


def create_numpy_inputs():
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP! ")
    decoder_inputs = tokenizer("It means you can")
    inputs = (
        np.array([inputs_dict["input_ids"]]).astype("int64"),
        None,
        np.array([decoder_inputs["input_ids"]]).astype("int64"),
    )
    return inputs
