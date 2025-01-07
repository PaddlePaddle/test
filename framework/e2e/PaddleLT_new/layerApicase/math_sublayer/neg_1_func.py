import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: neg_1
    api简介: 计算输入 x 的相反数并返回
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.neg(x,  )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([[0, 0, 0], [0, 0, 0]], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([[0, 0, 0], [0, 0, 0]]).astype('float32'), )
    return inputs

