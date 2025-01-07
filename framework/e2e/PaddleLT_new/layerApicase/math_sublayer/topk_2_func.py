import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: topk_2
    api简介: 沿着可选的 axis 查找topk最大或者最小的结果和结果所在的索引信息
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.topk(x,  k=3, axis=-2, largest=True, sorted=False, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([5, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([5, 3, 4, 4]).astype('float32'), )
    return inputs

