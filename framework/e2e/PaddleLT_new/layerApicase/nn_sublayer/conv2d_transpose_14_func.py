import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: conv2d_transpose_14
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
        out = paddle.nn.functional.conv2d_transpose(x,  weight=paddle.to_tensor(-1 + (1 - -1) * np.random.random([6, 1, 3, 3]).astype('float32'), dtype='float32', stop_gradient=False), stride=2, padding=[1, 0], dilation=1, groups=3, output_size=paddle.to_tensor([4, 6], dtype='int32', stop_gradient=False), data_format='NHWC', )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, 6), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 2, 2, 6]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 2, 2, 6]).astype('float32'), )
    return inputs

