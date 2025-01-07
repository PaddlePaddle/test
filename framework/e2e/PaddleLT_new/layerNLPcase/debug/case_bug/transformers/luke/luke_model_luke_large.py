import paddle
import numpy as np
from paddlenlp.transformers import LukeModel, LukeTokenizer

def LayerCase():
    """模型库中间态"""
    model = LukeModel.from_pretrained('luke-large')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 9), dtype=paddle.float32, stop_gradient=False),
        None,
        paddle.static.InputSpec(shape=(-1, 9), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 9), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 1), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 1, 30), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 1), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 1), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = LukeTokenizer.from_pretrained('luke-large')
    inputs_dict = tokenizer("Beyoncé lives in Los Angeles.", entity_spans=[(0, 7)], add_prefix_space=True)
    inputs = (
        paddle.to_tensor([inputs_dict['input_ids']], stop_gradient=False),
        None,
        paddle.to_tensor([inputs_dict['position_ids']], stop_gradient=False),
        paddle.to_tensor([inputs_dict['attention_mask']], stop_gradient=False),
        paddle.to_tensor([inputs_dict['entity_ids']], stop_gradient=False),
        paddle.to_tensor([inputs_dict['entity_position_ids']], stop_gradient=False),
        paddle.to_tensor([inputs_dict['entity_token_type_ids']], stop_gradient=False),
        paddle.to_tensor([inputs_dict['entity_attention_mask']], stop_gradient=False),
    )
    return inputs


def create_numpy_inputs():
    tokenizer = LukeTokenizer.from_pretrained('luke-large')
    inputs_dict = tokenizer("Beyoncé lives in Los Angeles.", entity_spans=[(0, 7)], add_prefix_space=True)
    inputs_aprac = {k:paddle.to_tensor([v]) for (k, v) in inputs_dict.items()}
    inputs = (
        np.array([inputs_dict['input_ids']]).astype("int64"),
        None,
        np.array([inputs_dict['position_ids']]).astype("int64"),
        np.array([inputs_dict['attention_mask']]).astype("int64"),
        np.array([inputs_dict['entity_ids']]).astype("int64"),
        np.array([inputs_dict['entity_position_ids']]).astype("int64"),
        np.array([inputs_dict['entity_token_type_ids']]).astype("int64"),
        np.array([inputs_dict['entity_attention_mask']]).astype("int64"),
    )
    return inputs
