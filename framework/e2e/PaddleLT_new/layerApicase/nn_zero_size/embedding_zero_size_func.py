import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: embedding_zero_size_class
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, data, weight):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = nn.functional.embedding(data, weight=weight, padding_idx=-1, sparse=False,)
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1), dtype=paddle.int32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (
        paddle.to_tensor(4 * np.random.random([0]).astype('int32'), dtype='int32', stop_gradient=False), 
        paddle.to_tensor(-1 + 2 * np.random.random([3, 2]).astype('float32'), dtype='float32', stop_gradient=False),
        )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (
        4 * np.random.random([0]).astype('int32'), 
        -1 + 2 * np.random.random([3, 2]).astype('float32'),
    )
    return inputs

