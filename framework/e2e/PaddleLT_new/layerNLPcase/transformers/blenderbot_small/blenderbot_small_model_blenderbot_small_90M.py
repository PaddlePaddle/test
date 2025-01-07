import paddle
import numpy as np
from paddlenlp.transformers import BlenderbotSmallModel, BlenderbotSmallTokenizer


def LayerCase():
    """模型库中间态"""
    model = BlenderbotSmallModel.from_pretrained("blenderbot_small-90M")
    return model


def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 15), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 15), dtype=paddle.float32, stop_gradient=False),
    )
    return inputspec


def create_tensor_inputs():
    tokenizer = BlenderbotSmallTokenizer.from_pretrained("blenderbot_small-90M")
    inputs_dict = tokenizer(
        "My friends are cool but they eat too many carbs.", return_attention_mask=True, return_token_type_ids=False
    )
    inputs = tuple(paddle.to_tensor([v], stop_gradient=False) for (k, v) in inputs_dict.items())
    return inputs


def create_numpy_inputs():
    tokenizer = BlenderbotSmallTokenizer.from_pretrained("blenderbot_small-90M")
    inputs_dict = tokenizer(
        "My friends are cool but they eat too many carbs.", return_attention_mask=True, return_token_type_ids=False
    )
    inputs = tuple(np.array([v]) for (k, v) in inputs_dict.items())
    return inputs
