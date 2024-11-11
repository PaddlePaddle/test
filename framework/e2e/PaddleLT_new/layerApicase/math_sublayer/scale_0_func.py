import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: scale_0
    api简介: 对输入Tensor进行缩放和偏置
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.scale(x,  scale=2.0, bias=4.0, bias_after_scale=True, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.int32, stop_gradient=True), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(-10, 10, [2, 3, 4, 4]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(-10, 10, [2, 3, 4, 4]).astype('int32'), )
    return inputs

