import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: label_smooth0
    api简介: 标签平滑正则化(LSR)
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, label, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.label_smooth(label,  prior_dist=paddle.to_tensor(0 + (1 - 0) * np.random.random([1, 8]).astype('float32'), dtype='float32', stop_gradient=False), epsilon=0.1, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(0 + (5 - 0) * np.random.random([2, 3, 8, 8]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (0 + (5 - 0) * np.random.random([2, 3, 8, 8]).astype('float32'), )
    return inputs

