import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: scatter_nd_base
    api简介: 根据 index ，将 updates 添加到一个新的张量中，从而得到输出的Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, index, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.scatter_nd(index,  updates=paddle.to_tensor(-1 + (1 - -1) * np.random.random([3, 9, 10]).astype('float32'), dtype='float32', stop_gradient=False), shape=[3, 5, 9, 10], )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, 2), dtype=paddle.int32, stop_gradient=True), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(0, 3, [3, 2]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(0, 3, [3, 2]).astype('int32'), )
    return inputs

