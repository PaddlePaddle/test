import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: index_select_0
    api简介: 沿着指定轴 axis 对输入 x 进行索引，取 index 中指定的相应项，创建并返回到一个新的Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, index, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.index_select(x, index, axis=1, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1,), dtype=paddle.int32, stop_gradient=True), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (
        paddle.to_tensor(-1 + (1 - -1) * np.random.random([128, 0, 3, 3]).astype('float32'), dtype='float32', stop_gradient=False), 
        paddle.to_tensor([0, 2], dtype='int32', stop_gradient=False), 
    )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (
        -1 + (1 - -1) * np.random.random([128, 0, 3, 3]).astype('float32'), 
        np.array([0, 2]).astype('int32'), 
    )
    return inputs

