import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: grid_sample_base
    api简介: 基于flow field网格的对输入X进行双线性插值采样
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.grid_sample(x,  grid=paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 8, 9, 2]).astype('float32'), dtype='float32', stop_gradient=False), mode='bilinear', padding_mode='zeros', align_corners=True, )
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
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 8, 8]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 8, 8]).astype('float32'), )
    return inputs

