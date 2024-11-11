import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: moveaxis_1
    api简介: 将输入Tensor x 的轴从 source 位置移动到 destination 位置，其他轴按原来顺序排布
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.moveaxis(x,  source=0, destination=1, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.int32, stop_gradient=True), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(-2, 2, [4, 2]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(-2, 2, [4, 2]).astype('int32'), )
    return inputs

