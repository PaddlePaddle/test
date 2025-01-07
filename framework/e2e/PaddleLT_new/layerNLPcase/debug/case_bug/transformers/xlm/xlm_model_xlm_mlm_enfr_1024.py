import paddle
import numpy as np
from paddlenlp.transformers import XLMModel, XLMTokenizer

def LayerCase():
    """模型库中间态"""
    model = XLMModel.from_pretrained('xlm-mlm-enfr-1024')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 16), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 16), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-tlm-xnli15-1024')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", lang="en")
    inputs = (
        paddle.to_tensor([inputs_dict['input_ids']], stop_gradient=False),
        paddle.ones_like(inputs_dict['input_ids']) * tokenizer.lang2id["en"],
    )
    return inputs


def create_numpy_inputs():
    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-tlm-xnli15-1024')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", lang="en")
    inputs = (
        np.array([inputs_dict["input_ids"]]).astype("int64"),
        np.ones((1, 16)).astype("int64") * 4,
    )
    return inputs
