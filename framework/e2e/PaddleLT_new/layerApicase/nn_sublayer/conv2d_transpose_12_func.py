import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: conv2d_transpose_12
    api简介: 2维反卷积
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.conv2d_transpose(x,  weight=paddle.to_tensor(-1 + (1 - -1) * np.random.random([3, 1, 5, 5]).astype('float32'), dtype='float32', stop_gradient=False), stride=1, padding='VALID', data_format='NCHW', dilation=1, output_padding=0, groups=1, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, 3, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 8, 8]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 8, 8]).astype('float32'), )
    return inputs

