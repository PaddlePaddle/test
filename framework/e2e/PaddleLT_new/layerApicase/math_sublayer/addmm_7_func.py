import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: addmm_7
    api简介: 计算x和y的乘积，将结果乘以标量alpha，再加上input与beta的乘积，得到输出
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, input, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.addmm(input, x, y,  alpha=3.3, beta=-3.3, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([1, 1]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-10 + (10 - -10) * np.random.random([5, 4]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-10 + (10 - -10) * np.random.random([4, 5]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([1, 1]).astype('float32'), -10 + (10 - -10) * np.random.random([5, 4]).astype('float32'), -10 + (10 - -10) * np.random.random([4, 5]).astype('float32'), )
    return inputs

