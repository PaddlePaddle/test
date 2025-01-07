import paddle
import numpy as np
from paddlenlp.transformers import UnifiedTransformerModel, UnifiedTransformerTokenizer

def LayerCase():
    """模型库中间态"""
    model = UnifiedTransformerModel.from_pretrained('plato-mini')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 15), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, 15), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = UnifiedTransformerTokenizer.from_pretrained('plato-mini')
    inputs_dict = tokenizer.dialogue_encode("我爱祖国", return_tensors=True, is_split_into_words=False)
    inputs = tuple(paddle.to_tensor([v], stop_gradient=False) for (k, v) in inputs_dict.items())
    return inputs


def create_numpy_inputs():
    tokenizer = UnifiedTransformerTokenizer.from_pretrained('plato-mini')
    inputs_dict = tokenizer.dialogue_encode("我爱祖国", return_tensors=True, is_split_into_words=False)
    inputs = tuple(np.array(v) for (k, v) in inputs_dict.items())
    return inputs
