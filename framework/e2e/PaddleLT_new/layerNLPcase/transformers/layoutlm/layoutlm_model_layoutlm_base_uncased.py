import paddle
import numpy as np
from paddlenlp.transformers import LayoutLMModel, LayoutLMTokenizer

def LayerCase():
    """模型库中间态"""
    model = LayoutLMModel.from_pretrained('layoutlm-base-uncased')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 13), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 13, 4), dtype=paddle.int64, stop_gradient=False),
        # paddle.static.InputSpec(shape=(-1, 3, 224, 224), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 13), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = LayoutLMTokenizer.from_pretrained('layoutlm-base-uncased')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    inputs = (
        paddle.to_tensor([inputs_dict['input_ids']], stop_gradient=False),
        paddle.to_tensor(np.random.random((1, 13, 4)).astype("int64"), stop_gradient=False),
        # paddle.to_tensor(np.random.random((1, 3, 224, 224)), stop_gradient=False),
        paddle.to_tensor([inputs_dict['token_type_ids']], stop_gradient=False),
    )
    return inputs


def create_numpy_inputs():
    tokenizer = LayoutLMTokenizer.from_pretrained('layoutlm-base-uncased')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    inputs = (
        np.array([inputs_dict['input_ids']]),
        np.random.random((1, 13, 4)).astype("int64"),
        # np.random.random((1, 3, 224, 224)),
        np.array([inputs_dict['token_type_ids']]),
    )
    return inputs
