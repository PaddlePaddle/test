import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: less_than_2
    api简介: 逐元素地返回 x<y 的逻辑值，相同位置前者输入小于后者输入则返回True，否则返回False
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.less_than(x, y,  )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([1, 2]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-10 + (10 - -10) * np.random.random([2, 2]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([1, 2]).astype('float32'), -10 + (10 - -10) * np.random.random([2, 2]).astype('float32'), )
    return inputs

