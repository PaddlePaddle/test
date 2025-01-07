import paddle
import numpy as np
from paddlenlp.transformers import BlenderbotModel, BlenderbotTokenizer


def LayerCase():
    """模型库中间态"""
    model = BlenderbotModel.from_pretrained("blenderbot-3B")
    return model


def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 15), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 15), dtype=paddle.float32, stop_gradient=False),
    )
    return inputspec


def create_tensor_inputs():
    tokenizer = BlenderbotTokenizer.from_pretrained("blenderbot-3B")
    inputs_dict = tokenizer(
        "My friends are cool but they eat too many carbs.", return_attention_mask=True, return_token_type_ids=False
    )
    inputs = tuple(paddle.to_tensor([v], stop_gradient=False) for (k, v) in inputs_dict.items())
    return inputs


def create_numpy_inputs():
    tokenizer = BlenderbotTokenizer.from_pretrained("blenderbot-3B")
    inputs_dict = tokenizer(
        "My friends are cool but they eat too many carbs.", return_attention_mask=True, return_token_type_ids=False
    )
    inputs = tuple(np.array([v]) for (k, v) in inputs_dict.items())
    return inputs
