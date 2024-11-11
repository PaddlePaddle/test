import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: less_equal_base
    api简介: 逐元素地返回 x<=y 的逻辑值，相同位置前者输入小于等于后者输入则返回True，否则返回False
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.less_equal(x, y,  )
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
    inputs = (paddle.to_tensor([[2.0, 1.0, -3.5], [-2.7, 1.5, 3], [0, 4.1, 8.6]], dtype='float32', stop_gradient=False), paddle.to_tensor([[-2.0, 1.1, -3.5], [-2.5, 1.5, 3.5], [0.5, 4.2, 8.3]], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([[2.0, 1.0, -3.5], [-2.7, 1.5, 3], [0, 4.1, 8.6]]).astype('float32'), np.array([[-2.0, 1.1, -3.5], [-2.5, 1.5, 3.5], [0.5, 4.2, 8.3]]).astype('float32'), )
    return inputs

