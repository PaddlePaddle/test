import paddle
import numpy as np
from paddlenlp.transformers import LayoutLMForTokenClassification
from paddlenlp.transformers import LayoutLMTokenizer

def LayerCase():
    """模型库中间态"""
    model = LayoutLMForTokenClassification.from_pretrained('layoutlm-base-uncased', num_classes=2)
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 13), dtype=paddle.float32, stop_gradient=False),
        None,
        None,
        paddle.static.InputSpec(shape=(-1, 13), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = LayoutLMTokenizer.from_pretrained('layoutlm-base-uncased')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", return_tensors="pd")
    inputs = (
        paddle.to_tensor(inputs_dict['input_ids'], stop_gradient=False),
        None,
        None,
        paddle.to_tensor(inputs_dict['token_type_ids'], stop_gradient=False),
    )
    return inputs


def create_numpy_inputs():
    tokenizer = LayoutLMTokenizer.from_pretrained('layoutlm-base-uncased')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", return_tensors="pd")
    inputs = (
        np.array(inputs_dict['input_ids']),
        None,
        None,
        np.array(inputs_dict['token_type_ids']),
    )
    return inputs
