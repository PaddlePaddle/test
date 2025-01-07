import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: unsqueeze_0
    api简介: 向输入Tensor的Shape中一个或多个位置（axis）插入尺寸为1的维度
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.unsqueeze(x,  axis=-1, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(3, 4, 5, 5), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([3, 4, 5, 5]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([3, 4, 5, 5]).astype('float32'), )
    return inputs

