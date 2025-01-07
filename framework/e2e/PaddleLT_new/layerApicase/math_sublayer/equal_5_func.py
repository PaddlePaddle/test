import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: equal_5
    api简介: 逐元素比较x和y是否相等，相同位置的元素相同则返回True，否则返回False
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.equal(x, y,  )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.bool, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1,), dtype=paddle.bool, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([[True, False, True], [True, False, True], [True, False, True]], dtype='bool', stop_gradient=False), paddle.to_tensor([True, False, True], dtype='bool', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([[True, False, True], [True, False, True], [True, False, True]]).astype('bool'), np.array([True, False, True]).astype('bool'), )
    return inputs

