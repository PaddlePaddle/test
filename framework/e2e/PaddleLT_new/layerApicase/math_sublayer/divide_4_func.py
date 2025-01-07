import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: divide_4
    api简介: 输入 x 与输入 y 逐元素相除，并将各个位置的输出元素保存到返回结果中
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.divide(x, y,  )
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
    inputs = (paddle.to_tensor([[-3], [-11], [-2]], dtype='float32', stop_gradient=False), paddle.to_tensor([[-1], [2], [1]], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([[-3], [-11], [-2]]).astype('float32'), np.array([[-1], [2], [1]]).astype('float32'), )
    return inputs

