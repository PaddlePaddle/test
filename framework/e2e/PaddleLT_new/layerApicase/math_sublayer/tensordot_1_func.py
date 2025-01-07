import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: tensordot_1
    api简介: 张量缩并运算（Tensor Contraction），即沿着axes给定的多个轴对两个张量对应元素的乘积进行加和操作
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.tensordot(x, y,  axes=1, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(3, 4, 3, 4), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(4, 4, 3, 4), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([3, 4, 3, 4]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-4 + (1 - -4) * np.random.random([4, 4, 3, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([3, 4, 3, 4]).astype('float32'), -4 + (1 - -4) * np.random.random([4, 4, 3, 4]).astype('float32'), )
    return inputs

