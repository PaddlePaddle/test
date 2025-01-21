import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: smooth_l1_loss_zero_size_class
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.smooth_l1_loss(
            x, y,
            reduction='mean',
            delta=1.0,
        )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, ), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, -1, ), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (
        paddle.to_tensor(-1 + (1 - -1) * np.random.random([10, 0]).astype('float32'), dtype='float32', stop_gradient=False), 
        paddle.to_tensor(-1 + (1 - -1) * np.random.random([10, 0]).astype('float32'), dtype='int64', stop_gradient=False),
    )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (
        -1 + (1 - -1) * np.random.random([10, 0]).astype('float32'), 
        -1 + (1 - -1) * np.random.random([10, 0]).astype('int64'), 
    )
    return inputs

