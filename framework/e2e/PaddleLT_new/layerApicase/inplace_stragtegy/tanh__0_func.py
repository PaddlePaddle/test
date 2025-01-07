import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: tanh__0
    api简介: tanh 激活函数, 对输入 x 采用 Inplace 策略
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.tanh_(x,  )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(2, 3, 4, 4), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-100 + (-1 - -100) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=True), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-100 + (-1 - -100) * np.random.random([2, 3, 4, 4]).astype('float32'), )
    return inputs

